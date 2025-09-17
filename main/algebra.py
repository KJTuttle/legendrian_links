from collections import Counter
import numpy as np
import pickle
import sympy
import utils
import polynomials

LOG = utils.get_logger(__name__)
PAGE_VAR = sympy.Symbol('r', commutative=True)
FILTRATION_VAR = sympy.Symbol('p', commutative=True)
DEGREE_VAR = sympy.Symbol('t', commutative=True)


class Differential(object):
    """Differential of a sympy symbol."""

    def __init__(self, expression, coeff_mod=2):
        self.coeff_mod = coeff_mod
        # store expression reduced modulo coeff_mod (so display and downstream agree)
        self.expression = self._reduce_mod(expression) if coeff_mod != 0 else sympy.sympify(expression)


    def __repr__(self):
        # Always render reduced form when working over a finite field
        if self.coeff_mod != 0:
            return str(self._reduce_mod(self.expression))
        return str(self.expression)

    def is_linear(self):
        """Returns True or False is the expression is linear"""
        return polynomials.is_linear(self.expression)

    def linearize(self, subs):
        """Return linearized differential"""
        return self.bilinearize(subs, subs)

    def bilinearize(self, subs_1, subs_2):
        """Return bilinearized differential using substitutions"""
        poly = sympy.sympify(self.expression)
        output = 0
        if not poly.is_number:
            # override use of defaultdict
            coeff_dict = dict(poly.as_coefficients_dict())
            output_summands = []
            for monom, coeff in coeff_dict.items():
                if monom.is_symbol:
                    output_summands.append(coeff * monom)
                elif not monom.is_number:
                    args = monom.args
                    # TODO: The reason that this works is because in the app an x^**2 term can never appear
                    # in a differential. This method is not applicable in abstract algebraic settings :(
                    for i in range(len(args)):
                        factor_1 = utils.prod([s.subs(subs_1) for s in args[:i]])
                        linear_term = args[i]
                        factor_2 = utils.prod([s.subs(subs_2) for s in args[i+1:]])
                        summand = coeff * factor_1 * linear_term * factor_2
                        output_summands.append(summand)
            output = sum(output_summands)
        return Differential(expression=output, coeff_mod=self.coeff_mod)
    
    def _reduce_mod(self, expr):
        poly = sympy.sympify(expr)
        # constant term
        if poly.is_number:
            return sympy.Integer(int(poly) % self.coeff_mod)
        # reduce each coefficient mod p
        coeff_dict = dict(poly.as_coefficients_dict())
        out = 0
        for monom, coeff in coeff_dict.items():
            try:
                # integers
                if getattr(coeff, 'q', 1) == 1:
                    c = int(coeff) % self.coeff_mod
                else:
                    # rationals a/b -> a * b^{-1} mod p (rare here)
                    a = int(coeff.p)
                    b = int(coeff.q)
                    inv_b = utils.num_inverse(b, self.coeff_mod)
                    c = (a * inv_b) % self.coeff_mod
            except Exception:
                c = int(sympy.Integer(coeff) % self.coeff_mod)
            if c != 0:
                out += c * monom
        return sympy.sympify(out)

class Matrix(object):
    """Attributes:

    values: np.array
    n_rows: Int
    n_cols: Int
    coeff_mod: Consider as matrix with coeffs modulus this number
    ref: Row echelon form
    ref_q: Matrix such that ref_q * ref = self
    ref_q_inv: Inverse of ref_q
    ref_form: Is the matrix REF (row echelon form) or RREF (reduced row echeleon form)?
    """
    REF = 'REF'
    RREF = 'RREF'

    def __init__(self, values, coeff_mod=0):
        self.values = np.array(values)

        self.n_rows = self.values.shape[0]
        self.n_cols = self.values.shape[1]
        self.coeff_mod = coeff_mod
        self._wierd_shape = False

        wrong_shape = len(self.values.shape) != 2
        wrong_shape |= self.values.shape[0] == 0
        if wrong_shape:
            self._init_wrong_shape()
            return

        if coeff_mod != 0:
            self.values %= self.coeff_mod
        self.ref_form = None
        self.ref = None
        self.ref_q = None
        self.ref_q_inv = None
        self.ref_free_indices = None
        self.ref_pivot_indices = None
        self._kernel = None
        self._rank_im = None
        self._rank_ker = None
        self._det = None

    def __repr__(self):
        return str(self.values)

    def __mul__(self, other):
        """Allows us to use X * Y with X a matrix and Y a matrix or array"""
        if isinstance(other, Matrix):
            return self.multiply(other)
        return self.multiply_array(other)

    def _init_wrong_shape(self):
        self._wierd_shape = True
        if self.values.shape[0] == 0:
            self.ref = self
            self.ref_q = self
            self.ref_q_inv = self
            self._rank_im = 0
            self._rank_ker = self.n_cols
            self._det = 0
            self.ref_pivot_indices = list()
            self.ref_free_indices = list(range(self.n_cols))
            return
        else:
            raise ValueError(f'Matrix has unfriendly shape {self.values.shape}')

    def delete(self, rows=[], cols=[]):
        """Return the Matrix obtained by deleting rows and columns"""
        return Matrix(np.delete(np.delete(self.values, rows, 0), cols, 1), coeff_mod=self.coeff_mod)

    def multiply(self, other):
        """Multiply self with another Matrix. Must have appropriate dimensions and have the same coeff modulus.

        :param other: Other instance of Matrix
        :return: Matrix
        """
        if not isinstance(other, Matrix):
            raise ValueError(f"Trying to multiply matrix with {other.__class__}")
        if not self.coeff_mod == other.coeff_mod:
            raise ValueError(f"Trying to multiply matrix having coeff_modulus={self.coeff_mod} "
                             f"with other having coeff_modulus={other.coeff_mod}")
        values = np.dot(self.values, other.values)
        return Matrix(values=values, coeff_mod=self.coeff_mod)

    def multiply_array(self, array):
        """Multiple self with a 1-d array.

        :param array: 1-d array (list or numpy array)
        :return: 1-d numpy array
        """
        output = np.dot(self.values, array)
        if self.coeff_mod != 0:
            output %= self.coeff_mod
        return output

    def rank_im(self):
        if self.n_rows == 0 or self.n_cols == 0:
            return 0
        if self._rank_im is None:
            self.set_row_echelon()
        return self._rank_im

    def rank_ker(self):
        if self.n_cols == 0:
            return 0
        if self.n_rows == 0:
            return self.n_cols
        if self._rank_ker is None:
            self.set_row_echelon()
        return self._rank_ker

    def set_row_echelon(self):
        """Set ref, ref_q, ref_q_inv, rank_im, rank_ker attributes for self. Sets self.ref_form = self.REF

        :return: None
        """
        if self._wierd_shape:
            return
        ref = self.values.copy()
        ref_q = np.identity(self.n_rows)
        ref_q_inv = np.identity(self.n_rows)
        row_num = 0
        col_num = 0
        while row_num < self.n_rows:
            while col_num < self.n_cols and not (np.any(ref[row_num:, col_num])):
                col_num += 1
            if col_num == self.n_cols:
                break
            ref, ref_q, ref_q_inv = self._row_reduce(ref, ref_q, ref_q_inv, row_num, col_num)
            row_num += 1
        self.ref = Matrix(ref, coeff_mod=self.coeff_mod)
        self.ref.ref_form = self.REF
        self.ref_q = Matrix(ref_q, coeff_mod=self.coeff_mod)
        self.ref_q_inv = Matrix(ref_q_inv, coeff_mod=self.coeff_mod)
        self._rank_im = row_num
        self._rank_ker = self.n_cols - row_num
        # need to set pivots and free indices
        pivot_indices = list()
        free_indices = list()
        loc_row = 0
        loc_col = 0
        while loc_col < self.n_cols:
            pivot = False
            if self.ref.values[loc_row, loc_col] == 0:
                free_indices.append(loc_col)
            elif loc_row not in [x[0] for x in pivot_indices]:
                pivot_indices.append([loc_row, loc_col])
                pivot = True
            loc_col += 1
            if pivot and loc_row < self.n_rows - 1:
                loc_row += 1
        self.ref_pivot_indices = pivot_indices
        self.ref_free_indices = free_indices
        self.ref_form = self.REF
        if self.n_rows == self.n_cols:
            det = utils.prod([self.ref.values[k, k] for k in range(self.n_cols)])
            if self.coeff_mod != 0:
                det %= self.coeff_mod
            self._det = det

    def set_red_row_echelon(self):
        """Override the ref variables with variables associated to reduced row echelon form,
        Sets self.ref_form = self.RREF

        :return: None
        """
        if self._wierd_shape:
            return
        if self.ref is None:
            self.set_row_echelon()
        ref, ref_q, ref_q_inv = self.ref.values, self.ref_q.values, self.ref_q_inv.values
        for row_num, col_num in self.ref_pivot_indices:
            if self.coeff_mod != 2:
                ref, ref_q, ref_q_inv = self._rref_normalize(ref, ref_q, ref_q_inv, row_num, col_num)
            ref, ref_q, ref_q_inv = self._rref_column_clean(ref, ref_q, ref_q_inv, row_num, col_num)
        self.ref = Matrix(ref, coeff_mod=self.coeff_mod)
        self.ref.ref_form = self.RREF
        self.ref_q = Matrix(ref_q, coeff_mod=self.coeff_mod)
        self.ref_q_inv = Matrix(ref_q_inv, coeff_mod=self.coeff_mod)
        self.ref_form = self.RREF

    def kernel(self):
        """Sets and returns the kernel as a list of column vectors"""
        if self._kernel is not None:
            return self._kernel
        if self.ref_form != self.RREF:
            self.set_red_row_echelon()
        # we can compute the span of the kernel using the free variable columns
        kernel = list()
        # sorted pivot indices with largest row coming first
        pivot_indices = sorted(self.ref_pivot_indices, key=lambda t: t[0], reverse=True)
        for i in self.ref_free_indices:
            free_column = self.ref.values[:, i]
            v = np.zeros(self.n_cols)
            v[i] = 1
            relevant_pivots = [[r, c] for r, c in pivot_indices if c < i]
            for r, c in relevant_pivots:
                coeff = -free_column[r]
                v[c] = coeff
                free_column += coeff*self.ref.values[:, c]
            if self.coeff_mod != 0:
                v %= self.coeff_mod
            kernel.append(v)
        self._kernel = kernel
        return self._kernel

    def is_square(self):
        """Return True or False for if the matrix is square"""
        return self.n_rows == self.n_cols

    def det(self):
        """Return determinant of matrix as element of Z/coeff_mod"""
        if self.n_rows != self.n_cols:
            return ValueError("Trying to compute det of non-square matrix")
        if self._det is None:
            self.set_row_echelon()
        return self._det

    def get_inverse(self):
        """Return inverse if it exists, otherwise raise ValueError"""
        if self.n_rows != self.n_cols:
            return ValueError("Trying to invert non-square matrix")
        if self.det() == 0:
            raise ValueError(f"Trying to invert non-invertible matrix,\n{self}")
        return self.ref_q_inv

    def _row_reduce(self, ref, ref_q, ref_q_inv, k, l):
        while np.any(ref[k + 1:, l]):
            ref, ref_q, ref_q_inv = self._ref_swap(ref, ref_q, ref_q_inv, k, l)
            ref, ref_q, ref_q_inv = self._ref_addition(ref, ref_q, ref_q_inv, k, l)
        if self.coeff_mod != 0:
            ref %= self.coeff_mod
            ref_q %= self.coeff_mod
            ref_q_inv %= self.coeff_mod
        return ref, ref_q, ref_q_inv

    def _ref_swap(self, ref, ref_q, ref_q_inv, row_num, col_num):
        (a, i) = self._smallest_nonzero_index(ref[:, col_num], row_num)
        ref[[i, row_num], :] = ref[[row_num, i], :]
        ref_q_inv[[i, row_num], :] = ref_q_inv[[row_num, i], :]  # row swap
        ref_q[:, [i, row_num]] = ref_q[:, [row_num, i]]  # column swap
        if self.coeff_mod != 0:
            ref %= self.coeff_mod
            ref_q %= self.coeff_mod
            ref_q_inv %= self.coeff_mod
        return ref, ref_q, ref_q_inv

    @staticmethod
    def _smallest_nonzero_index(v, k):
        # TODO: This pattern is bad
        try:
            alpha = min(abs(v[k:][np.nonzero(v[k:])]))
            i = min(i for i in range(k, len(v)) if abs(v[i]) == alpha)
            return alpha, i
        except:
            return 0, np.nan  # minNonZero causes this sometimes

    def _ref_addition(self, ref, ref_q, ref_q_inv, row_num, col_num):
        for i in range(row_num + 1, ref_q.shape[0]):
            coeff = (ref[i, col_num] // ref[row_num, col_num])
            if self.coeff_mod != 0:
                coeff %= self.coeff_mod
            # row add i,k,-q
            ref[i] += (-coeff * ref[row_num])
            ref_q_inv[i] += (-coeff * ref_q_inv[row_num])  # row add
            ref_q[:, row_num] += (coeff * ref_q[:, i])  # column add (note i,k are switched)
            if self.coeff_mod != 0:
                ref[i] %= self.coeff_mod
                ref_q_inv[i] %= self.coeff_mod
                ref_q[:, row_num] %= self.coeff_mod
        if self.coeff_mod != 0:
            ref %= self.coeff_mod
            ref_q %= self.coeff_mod
            ref_q_inv %= self.coeff_mod
        return ref, ref_q, ref_q_inv

    def _rref_normalize(self, ref, ref_q, ref_q_inv, row_num, col_num):
        coeff = ref[row_num, col_num]
        coeff_inv = utils.num_inverse(coeff, self.coeff_mod)
        # divide ref and ref_q_inv row by coefficient at the pivot
        ref[row_num, :] = coeff_inv * ref[row_num, :]
        ref_q_inv[row_num, :] = coeff_inv * ref_q_inv[row_num, :]
        # multiple ref_q column by coefficient at the pivot
        ref_q[:, row_num] = coeff * ref_q[:, row_num]
        if self.coeff_mod != 0:
            ref %= self.coeff_mod
            ref_q %= self.coeff_mod
            ref_q_inv %= self.coeff_mod
        return ref, ref_q, ref_q_inv

    def _rref_column_clean(self, ref, ref_q, ref_q_inv, row_num, col_num):
        # add rows to eliminate entries above the specified pivot entry
        if not ref[row_num, col_num] == 1:
            raise ValueError(f"ref[row_num, col_num] = {ref[row_num, col_num]} != 1")
        if row_num == 0:
            return ref, ref_q, ref_q_inv
        for i in range(0, row_num):
            coeff = ref[i, col_num]
            ref[i, :] -= coeff * ref[row_num, :]
            ref_q_inv[i, :] -= coeff * ref_q_inv[row_num, :]
            ref_q[:, row_num] += coeff * ref_q[:, i]
        if self.coeff_mod != 0:
            ref %= self.coeff_mod
            ref_q %= self.coeff_mod
            ref_q_inv %= self.coeff_mod
        return ref, ref_q, ref_q_inv


class LinearMap(object):
    """Interface to matrices for linear polynomial expressions"""

    def __init__(self, coeff_dict, range_symbols, coeff_mod=2):
        """
        :param coeff_dict: Dict of the form {symbol: sympy expression}
        :param range_symbols: Symbols of the target.
        :param coeff_mod: Coefficient modulus to use
        """
        self.coeff_dict = coeff_dict
        self.range_symbols = range_symbols
        self.coeff_mod = coeff_mod
        self._set_dims()
        self._set_domain_symbols()
        self._set_matrix()
        self._kernel = None

    def __repr__(self):
        return str(self.coeff_dict)

    def rank_ker(self):
        return self.matrix.rank_ker()

    def rank_im(self):
        return self.matrix.rank_im()

    def array_to_domain_expression(self, array):
        array_len = len(array)
        n_symbols = len(self.domain_symbols)
        if not array_len == n_symbols:
            raise ValueError(f"Trying to get expression for len(array)={array_len} from {n_symbols} symbols")
        v_out = 0
        for i in range(self.dim_domain):
            v_out += array[i] * self.domain_symbols[i]
        return v_out

    def kernel(self):
        if self._kernel is None:
            mat_ker = self.matrix.kernel()
            output = list()
            for v in mat_ker:
                output.append(self.array_to_domain_expression(v))
            self._kernel = output
        return self._kernel

    def _set_dims(self):
        self.dim_domain = len(self.coeff_dict.keys())
        self.dim_range = len(self.range_symbols)

    def _set_domain_symbols(self):
        """Encode input and output symbols as `onehot` vectors so that they could be turned into a matrix"""
        self.domain_symbols = sorted(list(self.coeff_dict.keys()), key=str)

    def _set_matrix(self):
        # The dtype should be essential to avoid float arithmetic errors
        values = np.zeros((len(self.range_symbols), len(self.domain_symbols)), dtype=np.integer)
        for i in range(len(self.domain_symbols)):
            for j in range(len(self.range_symbols)):
                temp_subs = {s: 0 if s != self.range_symbols[j] else 1 for s in self.range_symbols}
                value = sympy.sympify(self.coeff_dict[self.domain_symbols[i]]).subs(temp_subs)
                values[j][i] = value
        self.matrix = Matrix(values=values, coeff_mod=self.coeff_mod)


class DGBase(object):
    """Base class for differential graded objects which are instantiated using sympy expressions for differentials"""

    def __init__(self, gradings, differentials, filtration_levels=None, coeff_mod=0, grading_mod=0):
        """
        :param symbols: Dictionary mapping generator -> grading.
        :param differentials: Dictionary mapping generator -> polynomial.
        :param coeff_mod: m when using Z/mZ coeffs. Must be zero or prime.
        :param grading_mod: m when using Z/mZ grading.
        """
        self.gradings = gradings
        self.symbols = self.gradings.keys()
        self.differentials = differentials
        self.filtration_levels = filtration_levels
        self.coeff_mod = coeff_mod
        self.grading_mod = grading_mod
        self._verify_init_args()
        self._normalize_differentials_to_mod()
        self._correct_gradings()
        self._correct_filtration_levels()
        self._set_min_max_gradings()

    def get_generators_by_grading(self, grading):
        return [k for k, v in self.gradings.items() if v == grading]

    def get_generators_by_filtration(self, filtration):
        if self.filtration_levels is None:
            raise ValueError("Trying to access filtration levels where there are none")
        return [k for k, v in self.filtration_levels.items() if v == filtration]

    def reduced(self, ceoff_mod, grading_mod):
        """Return instance of self with coefficients and gradings reduced by new moduli"""
        if self.coeff_mod % ceoff_mod != 0:
            raise ValueError(f"New coefficient modulus {ceoff_mod} doesn't divide old modulus {self.coeff_mod}")
        if self.grading_mod % grading_mod != 0:
            raise ValueError(f"New grading modulus {grading_mod} doesn't divide old modulus {self.grading_mod}")
        return self.__init__(
            gradings=self.gradings,
            differentials=self.differentials,
            coeff_mod=ceoff_mod,
            grading_mod=grading_mod
        )

    def _set_min_max_gradings(self):
        grading_values = set(self.gradings.values())
        self.max_grading = max(grading_values)
        self.min_grading = min(grading_values)

    def _correct_filtration_levels(self):
        if self.filtration_levels is None:
            self.filtration_levels = {k: 1 for k in self.gradings.keys()}
            self.max_filtration = 1
            return
        self.max_filtration = max(self.filtration_levels.values())

    def _verify_init_args(self):
        if self.coeff_mod != 0:
            if not sympy.isprime(self.coeff_mod):
                raise ValueError(f"Coefficient modulus {self.coeff_mod} is not 0 or prime")
        if self.symbols != self.differentials.keys():
            raise ValueError("generators don't match in symbols and keys")
        for g, d in self.differentials.items():
            if not isinstance(d, Differential):
                raise ValueError(f"Differential for {g} is not instance of class Differential.")
    def _normalize_differentials_to_mod(self):
        # Re-wrap differentials with this DGA’s coeff_mod so printing and algebra agree
        self.differentials = {
            g: Differential(d.expression, coeff_mod=self.coeff_mod)
            for g, d in self.differentials.items()
        }
    def _correct_gradings(self):
        if self.grading_mod != 0:
            corrected_gradings = {g: self.gradings[g] % self.grading_mod for g in self.gradings.keys()}
            self.gradings = corrected_gradings


class ChainComplex(DGBase):
    """Additive chain complex instantiated from sympy expressions for differentials.

    Use this class to access chain complex homotopy invariants.
    """

    def __init__(self, gradings, differentials, filtration_levels=None, coeff_mod=0, grading_mod=0):
        super(ChainComplex, self).__init__(
            gradings=gradings,
            differentials=differentials,
            filtration_levels=filtration_levels,
            coeff_mod=coeff_mod,
            grading_mod=grading_mod
        )
        self._verify_linear_differentials()
        self._set_matrix_cx()
        self._poincare_poly = None

    def rank_homology(self, grading):
        """:return: int rank of homology at specified grading degree"""
        return self.matrix_cx.rank_homology(grading)

    def poincare_poly(self):
        """:return: sympy expression for poincare polynomial"""
        if self._poincare_poly is None:
            self._poincare_poly = self.matrix_cx.poincare_poly()
        return self._poincare_poly

    def _verify_linear_differentials(self):
        for k, v in self.differentials.items():
            if not v.is_linear():
                raise ValueError(f"Trying to instantiate ChainComplex with non-linear differential {k}: {v}")

    def _set_matrix_cx(self):
        differentials = dict()
        ranks = dict()
        if self.grading_mod != 0:
            gradings = polynomials.finite_field_elements(self.grading_mod)
        else:
            gradings = range(self.min_grading, self.max_grading + 1)
        for i in gradings:
            i_min_1 = i - 1
            if self.grading_mod != 0:
                i_min_1 %= self.grading_mod
            range_symbols = [s for s in self.symbols if self.gradings[s] == i_min_1]
            domain_symbols = [s for s in self.symbols if self.gradings[s] == i]
            ranks[i] = len(domain_symbols)
            differentials[i] = LinearMap(
                coeff_dict={s: self.differentials[s].expression for s in domain_symbols},
                range_symbols=range_symbols,
                coeff_mod=self.coeff_mod,
            ).matrix
        self.matrix_cx = MatrixChainComplex(
            ranks=ranks,
            differentials=differentials,
            coeff_mod=self.coeff_mod,
            grading_mod=self.grading_mod
        )


class MatrixChainComplex(object):
    """Chain complex determined by a pair of dictionaries

    ranks[deg] -> number of generators of degree deg space
    differentials[deg] -> if ranks[deg] & ranks[deg - 1] are both non-zero, then gives differential as a Matrix
    """

    def __init__(self, ranks, differentials, coeff_mod=2, grading_mod=0):
        self.ranks = ranks
        self._set_chi()
        self.grading_mod = grading_mod
        self.coeff_mod = coeff_mod
        if self.coeff_mod != 0:
            if not sympy.isprime(self.coeff_mod):
                raise ValueError(f"Coefficient modulus {self.coeff_mod} is not 0 or prime")
        self.differentials = differentials
        self._validate_differentials()
        self.max_grading = max(ranks.keys())
        self.min_grading = min(ranks.keys())
        self._poincare_poly = None
        self._dim_kers = dict()
        self._dim_ims = dict()
        self.qrs_decomposition = None

    def poincare_poly(self):
        """:return: sympy expression for poincare polynomial"""
        if self._poincare_poly is not None:
            return self._poincare_poly
        output = 0
        if self.grading_mod != 0:
            gradings = polynomials.finite_field_elements(self.grading_mod)
        else:
            gradings = range(self.min_grading, self.max_grading + 1)
        poincare_chi = 0
        for grading in gradings:
            poincare_chi += self.rank_homology(grading) * ((-1)**grading)
            output += self.rank_homology(grading) * (DEGREE_VAR ** grading)
        if poincare_chi != self.chi:
            raise RuntimeError(f"chi={self.chi} computed from ranks "
                               f"not matching chi={poincare_chi} computed "
                               f"from poincare polynomial {output}\n"
                               f"self.grading_mod={self.grading_mod}\n"
                               f"ranks={self.ranks}\n"
                               f"_dim_kers={self._dim_kers}\n"
                               f"_dim_ims={self._dim_ims}\n"
                               f"differentials={self.differentials}")
        self._poincare_poly = output
        return self._poincare_poly

    def rank_im(self, grading):
        """:return: int rank of image of differential at specified grading degree"""
        if self.grading_mod != 0:
            grading %= self.grading_mod
        if grading in self._dim_ims.keys():
            return self._dim_ims[grading]
        if grading not in self.differentials.keys():
            self._dim_ims[grading] = 0
        else:
            self._dim_ims[grading] = self.differentials[grading].rank_im()
        return self._dim_ims[grading]

    def rank_ker(self, grading):
        """:return: int rank of kernel of differential at specified grading degree"""
        if self.grading_mod != 0:
            grading %= self.grading_mod
        if grading in self._dim_kers.keys():
            return self._dim_kers[grading]
        if grading not in self.differentials.keys():
            self._dim_kers[grading] = self.ranks.get(grading, 0)
        else:
            self._dim_kers[grading] = self.differentials[grading].rank_ker()
        return self._dim_kers[grading]

    def rank_homology(self, grading):
        """:return: int rank of homology at specified grading degree"""
        gr_plus_1 = grading + 1
        if self.grading_mod != 0:
            grading = grading % self.grading_mod
            gr_plus_1 %= self.grading_mod
        return self.rank_ker(grading) - self.rank_im(gr_plus_1)

    @utils.log_start_stop
    def set_qrs_decomposition(self):
        """Compute qrs decomposition of the chain complex. Requires complex to be Z-graded"""
        if self.grading_mod != 0:
            raise ValueError(f"Trying to compute qrs decomposition of MatrixChainComplex"
                             f" with grading_mod={self.grading_mod}")
        if self.qrs_decomposition is not None:
            return
        current_grading = self.max_grading
        qrs_decomposition = dict()
        dim_ims = dict()
        dim_kers = dict()
        previous_ref_q = None
        previous_im_rank = 0
        while current_grading >= self.min_grading:
            # current_matrix = M_n
            current_matrix = self.differentials.get(current_grading)
            if current_matrix is None or 0 in current_matrix.values.shape or current_matrix.n_cols == previous_im_rank:
                dim_ims[current_grading] = 0
                dim_kers[current_grading] = self.ranks.get(current_grading, 0)
                current_grading -= 1
                continue

            # Set current matrix to M_n Q_{n+1}
            if previous_ref_q is not None:
                current_matrix = current_matrix * previous_ref_q

            # Compute Q_n, R_n so that Q_n R_n = M_n Q_{n+1}.
            # Then throw out the first columns of M_n Q_{n+1} which correspond to
            # image of the previous del and compute RREF.
            # We have to add back in the thrown out columns to recover R_n
            truncated_matrix = Matrix(
                current_matrix.values[:, previous_im_rank:],
                coeff_mod=self.coeff_mod
            )
            truncated_matrix.set_red_row_echelon()
            dim_ims[current_grading] = truncated_matrix.rank_im()
            dim_kers[current_grading] = previous_im_rank + truncated_matrix.rank_ker()
            if previous_im_rank > 0:
                expanded_ref = Matrix(
                    np.concatenate(
                        (np.zeros((current_matrix.n_rows, previous_im_rank)), truncated_matrix.ref_q), axis=1),
                    coeff_mod=self.coeff_mod
                )
            else:
                expanded_ref = truncated_matrix.ref
            # check that the expanded ref has the same dimensions as the original matrix
            # the rediculous error message will help trouble shoot
            if current_matrix.values.shape != expanded_ref.values.shape:
                raise RuntimeError(f"current_matrix.values.shape={current_matrix.values.shape} != "
                                   f"expanded_ref.values.shape={expanded_ref.values.shape}.\n"
                                   f"previous_im_rank={previous_im_rank},\n"
                                   f"truncated_matrix.values.shape={truncated_matrix.values.shape}")
            # Update gradings
            qrs_decomposition[current_grading] = {
                "ref_q": truncated_matrix.ref_q,
                "ref_q_inv": truncated_matrix.ref_q_inv,
                "ref": expanded_ref
            }

            # compute the S_n matrix
            s_matrix_rows = []
            # kernels for the truncated R_n matrix, upgraded to have the correct dimension
            for v in truncated_matrix.kernel():
                s_matrix_rows.append(np.concatenate((np.zeros(previous_im_rank), v), axis=0))
            # vectors spanning image of the previous differential
            for i in range(previous_im_rank):
                s_matrix_rows.append(utils.one_hot_array(i, current_matrix.n_cols))
            # vectors for the pivot indices of the truncated R_n matrix
            r_n_pivots_cols = [ind[1] for ind in truncated_matrix.ref_pivot_indices]
            for col_num in r_n_pivots_cols:
                s_matrix_rows.append(utils.one_hot_array(col_num + previous_im_rank, current_matrix.n_cols))
            s_matrix = Matrix(s_matrix_rows, coeff_mod=self.coeff_mod)
            if not s_matrix.n_cols == current_matrix.n_cols:
                raise RuntimeError(f"s_matrix.n_cols = {s_matrix.n_cols} "
                                   f"!= current_matrix.n_cols = {current_matrix.n_cols}")
            if not s_matrix.n_rows == current_matrix.n_cols:
                raise RuntimeError(f"s_matrix.n_rows = {s_matrix.n_rows} "
                                   f"!= current_matrix.n_cols = {current_matrix.n_cols}")
            qrs_decomposition[current_grading]["s"] = s_matrix
            qrs_decomposition[current_grading]["s_inv"] = s_matrix.get_inverse()

            # set up for next iteration
            if current_grading - 1 in self.differentials.keys():
                previous_ref_q = truncated_matrix.ref_q
            current_grading -= 1
        self.qrs_decomposition = qrs_decomposition

    def _set_chi(self):
        self.chi = sum([v*((-1)**k) for k, v in self.ranks.items()])

    def _validate_differentials(self):
        for k, d in self.differentials.items():
            if not isinstance(d, Matrix):
                raise ValueError(f"Differential {d} is not a matrix")
            if not d.coeff_mod == self.coeff_mod:
                raise ValueError(f"Coeff_mods disagree {d.coeff_mod} != {self.coeff_mod}.")
        # Throw out matrices which have no rows
        self.differentials = {k: v for k, v in self.differentials.items() if v.n_rows != 0}
        for k, d in self.differentials.items():
            if d.n_cols != self.ranks[k]:
                raise ValueError(f"Differential at degree {k} "
                                 f"has n_cols={d.n_cols} != self.ranks[k] = {self.ranks[k]}")
            k_min_1 = k - 1
            if self.grading_mod != 0:
                k_min_1 %= self.grading_mod
            k_min_1_rank = self.ranks.get(k_min_1, 0)
            if d.n_rows != k_min_1_rank:
                raise ValueError(f"Differential at degree k={k} "
                                 f"has n_rows={d.n_rows} != self.ranks[{k_min_1}] = {k_min_1_rank}")


class SpectralSequence(DGBase):
    """Spectral sequence associated to a filtered chain complex. Only available for Z-gradings (grading_mod=0)"""

    def __init__(self, gradings, differentials, filtration_levels=None, coeff_mod=2, grading_mod=0):
        super(SpectralSequence, self).__init__(
            gradings=gradings,
            differentials=differentials,
            filtration_levels=filtration_levels,
            coeff_mod=coeff_mod,
            grading_mod=grading_mod
        )
        self._verify_z_grading()
        self._rank_homology = None
        self._poincare_poly = None
        self._matrices = None

    @utils.log_start_stop
    def poincare_poly(self):
        if self._poincare_poly is not None:
            return self._poincare_poly

        self._set_filtered_vars()
        self._set_filtered_differentials()
        self._set_initial_matrices()

        rank_homology = dict()
        poincare_poly = 0
        page_num = 0
        # ranks[(filt, deg)] = # variables with prescribed filtration level and degree
        # on page r, this will be the rank of E^{p}_{filt, deg}
        # We'll throw out the ones with zero rank
        page_ranks = {k: len(v) for k, v in self._filtered_vars.items()}
        page_ranks = {k: v for k, v in page_ranks.items() if v != 0}
        while page_num < self.max_filtration:
            shapes = {k: v.values.shape for k, v in self._matrices.items()}
            LOG.info(f"Computing page={page_num} of spectral sequence using \n"
                     f"ranks = {page_ranks},\n"
                     f"matrix shapes = {shapes}")
            self._verify_matrix_dimensions()
            page_data = list()
            untouched_indices = set(page_ranks.keys())
            while len(untouched_indices) > 0:
                # collect all of the data according to one line on the page and add it to page_data
                # starting with some random unanalyzed index
                filt, deg = untouched_indices.pop()
                cx_indices = [(filt, deg)]
                # Using a copy so we don't modify the list during iteration
                for f, d in untouched_indices.copy():
                    # (filt, deg) -> (filt - page_num, deg - 1) -> (filt - 2 (page_num), deg - 2)
                    # f = filt + (d - deg)(page_num)
                    if f == filt + (d - deg)*page_num:
                        cx_indices.append((f, d))
                        untouched_indices.remove((f, d))
                # throw out these variables as I want to reuse their names
                del filt
                del deg
                # sort the indices by their degrees (descending)
                cx_indices.sort(key=lambda pair: pair[1], reverse=True)
                LOG.info(f"Analyzing line on page_num={page_num} with indices {cx_indices}")

                # organize differentials and ranks by degree to set up matrix chain complex
                cx_differentials = {
                    loc_dom[1]: self._matrices[(loc_dom, loc_target)]
                    for loc_dom in cx_indices
                    for loc_target in cx_indices
                    if (loc_dom, loc_target) in self._matrices.keys()
                }
                cx_ranks = {deg: page_ranks[(filt, deg)] for filt, deg in cx_indices}
                page_data.append(
                    {
                        'cx': MatrixChainComplex(
                                ranks=cx_ranks,
                                differentials=cx_differentials,
                                coeff_mod=self.coeff_mod,
                                grading_mod=self.grading_mod
                            ),
                        'indices': cx_indices
                    }
                )
            # throw out the ranks and matrices which are used for the current page
            # don't need these for computation of future pages
            page_indices = list()
            for data in page_data:
                page_indices += data['indices']
            this_page_indices = [((filt, deg), (filt - page_num, deg-1)) for filt, deg in page_indices]
            self._matrices = {k: v for k, v in self._matrices.items() if k not in this_page_indices}

            # compute qrs decomposition of each complex in the page and
            # gather into transformation data for the next page
            page_q = dict()
            page_q_inv = dict()
            page_s = dict()
            page_s_inv = dict()
            LOG.info(f"Computed {len(page_data)} chain complexes on page_num={page_num}")
            new_ranks = dict()
            while len(page_data) > 0:
                data = page_data.pop()
                data['cx'].set_qrs_decomposition()
                for filt, deg in data['indices']:
                    # add contributions to the specseq homology ranks and poincare polynomial
                    betti = data['cx'].rank_homology(deg)
                    rank_homology[(page_num + 1, filt, deg)] = betti
                    new_ranks[(filt, deg)] = betti
                    poincare_poly += betti * (DEGREE_VAR ** deg) * (FILTRATION_VAR ** filt) * (PAGE_VAR ** page_num)
                    # add the transformation data going from
                    # (filt, deg) -> (filt - page_num * deg, deg - 1)
                    if deg in data['cx'].qrs_decomposition.keys():
                        page_q[(filt, deg)] = data['cx'].qrs_decomposition[deg]['ref_q']
                        page_q_inv[(filt, deg)] = data['cx'].qrs_decomposition[deg]['ref_q_inv']
                        page_s[(filt, deg)] = data['cx'].qrs_decomposition[deg]['s']
                        page_s_inv[(filt, deg)] = data['cx'].qrs_decomposition[deg]['s_inv']
            page_ranks = new_ranks

            # manual override prevents us from applying a bunch of unneeded transformations below
            if page_num == self.max_filtration - 1:
                break
            # update self.matrices by applying matrix mult and slicing only the relevant indices
            new_matrices = dict()
            for loc_dom, loc_target in self._matrices.keys():
                if page_ranks.get(loc_dom, 0) == 0 or page_ranks.get(loc_target, 0) == 0:
                    continue
                mat = self._matrices[(loc_dom, loc_target)]
                if mat is None or 0 in mat.values.shape:
                    continue
                # now replace mat with
                # S_{filt_target, deg - 1} * Q_{filt_target, deg}^{-1} * mat * Q_{filt, deg+1} * S_{filt, deg}^{-1}
                # and slice to get the first b^{r}_{filt_target, deg-1} rows and first b^{r}_{filt, deg} columns
                filt_dom, deg = loc_dom
                filt_target, deg_target = loc_target
                if not deg_target == deg - 1:
                    raise ValueError()
                try:
                    if (filt_dom, deg + 1) in page_q.keys():
                        mat *= page_q[(filt_dom, deg + 1)]
                    if (filt_dom, deg) in page_s_inv.keys():
                        mat *= page_s_inv[(filt_dom, deg)]
                    if (filt_target, deg) in page_q_inv.keys():
                        mat = page_q_inv[(filt_target, deg)] * mat
                    if (filt_target, deg-1) in page_s.keys():
                        mat = page_s[(filt_target, deg-1)] * mat
                except ValueError as e:
                    LOG.info(f"Failed at filt_dom, filt_target, deg = {filt_dom}, {filt_target}, {deg}")
                    raise e
                mat = mat.values[
                      :rank_homology.get((page_num + 1, filt_target, deg - 1), 0),
                      :rank_homology.get((page_num + 1, filt_dom, deg), 0),
                      ]
                new_matrices[(loc_dom, loc_target)] = Matrix(mat, coeff_mod=self.coeff_mod)
            self._matrices = new_matrices
            page_num += 1
        self._poincare_poly = poincare_poly
        return self._poincare_poly

    def _verify_z_grading(self):
        if self.grading_mod != 0:
            raise NotImplementedError("Spectral sequences only implemented for Z graded complexes")

    def _set_filtered_vars(self):
        filtered_vars = dict()
        for filt_level in range(1, self.max_filtration + 1):
            for deg in range(self.min_grading, self.max_grading + 1):
                fv = self.get_generators_by_filtration(filt_level)
                vars = [v for v in self.get_generators_by_grading(deg) if v in fv]
                filtered_vars[(filt_level, deg)] = sorted(vars, key=str)
        self._filtered_vars = filtered_vars

    def _set_filtered_differentials(self):
        """Sets filtered differentials as expressions"""
        diffs = dict()
        for filt_dom, deg in self._filtered_vars.keys():
            for filt_target in [f[0] for f in list(self._filtered_vars.keys()) if f[0] <= filt_dom]:
                non_target_subs = {k: 0 for k, v in self.filtration_levels.items() if v != filt_target}
                diffs[((filt_dom, deg), (filt_target, deg - 1))] = {
                    k: sympy.sympify(v.expression).subs(non_target_subs)
                    for k, v in self.differentials.items()
                    if k in self._filtered_vars[(filt_dom, deg)]
                }
        self._filtered_diffs = diffs

    def _set_initial_matrices(self):
        mats = dict()
        for k in self._filtered_diffs.keys():
            loc_dom, loc_target = k
            linear_map = LinearMap(
                coeff_dict=self._filtered_diffs[k],
                range_symbols=self._filtered_vars.get(loc_target, list()),
                coeff_mod=self.coeff_mod
            )
            mats[k] = linear_map.matrix
        self._matrices = mats

    def _verify_matrix_dimensions(self):
        ranks = dict()
        for k in self._matrices.keys():
            loc_dom, loc_target = k
            mat = self._matrices[k]
            if loc_dom not in ranks:
                ranks[loc_dom] = mat.n_cols
            else:
                if ranks[loc_dom] != mat.n_cols:
                    shapes = {k: v.values.shape for k, v in self._matrices.items() if k[0] == loc_dom}
                    raise ValueError(f"Matrices have mimatching dimensions for domain (filt, deg)={loc_dom}:\n"
                                     f"shapes={shapes}")
            if loc_target not in ranks:
                ranks[loc_target] = mat.n_rows
            else:
                if ranks[loc_target] != mat.n_rows:
                    shapes = {k: v.values.shape for k, v in self._matrices.items() if k[1] == loc_target}
                    raise ValueError(f"Matrices have mimatching dimensions for target (filt, deg)={loc_target}:\n"
                                     f"shapes={shapes}")


class DGA(DGBase):

    def __init__(self, gradings, differentials, filtration_levels=None, coeff_mod=0, grading_mod=0, spec_poly=False,
                 lazy_augs=False, lazy_bilin=False, aug_fill_na=None):
        super(DGA, self).__init__(
            gradings=gradings,
            differentials=differentials,
            filtration_levels=filtration_levels,
            coeff_mod=coeff_mod,
            grading_mod=grading_mod
        )
        if spec_poly and grading_mod != 0:
            raise ValueError(f"Spectral sequence computations only available when grading_mod=0 != {grading_mod}")
        self.spec_poly=spec_poly
        self.augmentations = None
        self.augmentations_compressed = None
        self.n_augs = None
        self.n_augs_compressed = None
        self.bilin_polys = dict()
        self.bilin_polys_dual = dict()
        self.lin_poly_list = None
        self.bilin_poly_list = None
        self.lazy_augs = lazy_augs
        self.lazy_bilin = lazy_bilin
        self.aug_fill_na = aug_fill_na
        if lazy_augs and not lazy_bilin:
            raise ValueError("DGA cannot autocompute bilinearized polys without autocomputing augmentations")

        # Variables needed to setup augs
        self._aug_vars_set = False
        self.aug_comm_symbols_to_symbols = None
        self.aug_symbols_to_comm_symbols = None
        self.aug_analysis_table = None
        self.aug_polys = None
        if not lazy_augs:
            self.set_aug_data()
            self.set_augmentations()
            if not lazy_bilin:
                self.set_all_bilin()

    def get_cylinder(self, spec_poly=False, lazy_augs=False, lazy_bilin=False, aug_fill_na=None):
        """Return Baues-Lemair cylinder"""

        symbols_l = {v: sympy.Symbol(v.name + 'L', commutative=False) for v in self.symbols}
        symbols_r = {v: sympy.Symbol(v.name + 'R', commutative=False) for v in self.symbols}
        symbols_h = {v: sympy.Symbol(v.name + 'H', commutative=False) for v in self.symbols}

        cyl_gradings = dict()
        for k, v in self.gradings.items():
            cyl_gradings[symbols_l[k]] = v
            cyl_gradings[symbols_r[k]] = v
            cyl_gradings[symbols_h[k]] = v + 1

        cyl_diffs = dict()
        for k, v in self.differentials.items():
            poly = sympy.sympify(v.expression)
            diff_l = 0
            diff_r = 0
            diff_h = symbols_r[k] - symbols_l[k]
            if poly.is_number:
                diff_l = poly
                diff_r = poly
            else:
                # override use of defaultdict
                coeff_dict = dict(poly.as_coefficients_dict())
                for monom, coeff in coeff_dict.items():
                    if monom == 1:
                        diff_l += coeff
                        diff_r += coeff
                    if monom.is_symbol:
                        diff_l += coeff * symbols_l[monom]
                        diff_r += coeff * symbols_r[monom]
                        diff_h += coeff * symbols_h[monom]
                    else:
                        args = monom.args
                        for i in range(len(args)):
                            diff_l += coeff * monom.subs(symbols_l)
                            diff_r += coeff * monom.subs(symbols_r)
                            factor_1 = utils.prod([s.subs(symbols_l) for s in args[:i]])
                            linear_term = args[i].subs(symbols_h)
                            factor_2 = utils.prod([s.subs(symbols_r) for s in args[i + 1:]])
                            diff_h += coeff * factor_1 * linear_term * factor_2
            cyl_diffs[symbols_l[k]] = Differential(expression=diff_l, coeff_mod=self.coeff_mod)
            cyl_diffs[symbols_r[k]] = Differential(expression=diff_r, coeff_mod=self.coeff_mod)
            cyl_diffs[symbols_h[k]] = Differential(expression=diff_h, coeff_mod=self.coeff_mod)
        return DGA(
            gradings=cyl_gradings,
            differentials=cyl_diffs,
            filtration_levels=self.filtration_levels,
            coeff_mod=self.coeff_mod,
            grading_mod=self.grading_mod,
            spec_poly=spec_poly,
            lazy_augs=lazy_augs, lazy_bilin=lazy_bilin,
            aug_fill_na=aug_fill_na
        )


    def pickle(self, file_name, compress=True):
        storage = dict()
        # Required for init
        storage['gradings'] = self.gradings
        storage['differentials'] = self.differentials
        storage['filtration_levels'] = self.filtration_levels
        storage['bilin_polys'] = self.bilin_polys
        storage['lin_poly_list'] = self.lin_poly_list
        storage['bilin_poly_list'] = self.bilin_poly_list
        storage['coeff_mod'] = self.coeff_mod
        storage['grading_mod'] = self.grading_mod
        # Data we want to set up
        storage['_aug_vars_set'] = self._aug_vars_set
        storage['aug_comm_symbols_to_symbols'] = self.aug_comm_symbols_to_symbols
        storage['aug_symbols_to_comm_symbols'] = self.aug_symbols_to_comm_symbols
        storage['aug_analysis_table'] = self.aug_analysis_table
        storage['aug_polys'] = self.aug_polys
        storage['augmentations_compressed'] = self.augmentations_compressed
        if compress:
            storage['augmentations'] = None
        else:
            storage['augmentations'] = self.augmentations
        with open(file_name, 'wb') as f:
            pickle.dump(storage, f)

    @classmethod
    def from_pickle(cls, file_name, decompress=False):
        with open(file_name, 'rb') as f:
            storage = pickle.load(f)
        dga = DGA(
            gradings=storage['gradings'],
            differentials=storage['differentials'],
            filtration_levels=storage['filtration_levels'],
            coeff_mod=storage['coeff_mod'],
            grading_mod=storage['grading_mod'],
            lazy_augs=True,
            lazy_bilin=True
        )
        dga._aug_vars_set = storage.get('_aug_vars_set')
        dga.aug_comm_symbols_to_symbols = storage.get('aug_comm_symbols_to_symbols')
        dga.aug_symbols_to_comm_symbols = storage.get('aug_symbols_to_comm_symbols')
        dga.aug_analysis_table = storage.get('aug_analysis_table')
        dga.aug_polys = storage.get('aug_polys')
        dga.bilin_polys = storage.get('bilin_polys')
        dga.lin_poly_list = storage.get('lin_poly_list')
        dga.bilin_poly_list = storage.get('bilin_poly_list')
        dga.augmentations_compressed = storage.get('augmentations_compressed')
        dga.augmentations = storage.get('augmentations')
        if decompress and dga.augmentations is None:
            dga.decompress_augmentations()
        return dga

    def get_verbose_subs_from_aug(self, aug):
        """Extend aug to all generators. This means setting non-zero graded generators to zero.

        :param aug: dict
        :return: dict
        """
        for s in self.symbols:
            if s not in aug.keys():
                aug[s] = 0
        return aug

    def get_bilin_differential(self, aug_1, aug_2):
        subs_1 = self.get_verbose_subs_from_aug(aug_1)
        subs_2 = self.get_verbose_subs_from_aug(aug_2)
        output = dict()
        for g, v in self.differentials.items():
            bi_diff = v.bilinearize(subs_1, subs_2)
            output[g] = bi_diff
        return output

    @utils.log_start_stop
    def set_all_bilin(self):
        bilin_counter = 0
        for i in range(self.n_augs):
            for j in range(self.n_augs):
                aug_1 = self.augmentations[i]
                aug_2 = self.augmentations[j]
                bilin_diff = self.get_bilin_differential(aug_1, aug_2)
                LOG.info(f"Computing bilinearized poincare poly for\n"
                         f"gradings={self.gradings}\n"
                         f"differentials={bilin_diff}")
                if self.spec_poly:
                    cx = SpectralSequence(
                        gradings=self.gradings,
                        filtration_levels=self.filtration_levels,
                        differentials=bilin_diff,
                        grading_mod=self.grading_mod,
                        coeff_mod=self.coeff_mod,
                    )
                else:
                    cx = ChainComplex(
                        gradings=self.gradings,
                        differentials=bilin_diff,
                        grading_mod=self.grading_mod,
                        coeff_mod=self.coeff_mod,
                    )
                self.bilin_polys[(i,j)] = cx.poincare_poly()
                bilin_counter += 1
                LOG.info(f"Computed {bilin_counter} of {self.n_augs ** 2} bilinearized Poincare polys so far")
        self.lin_poly_list = sorted(
            utils.unique_elements([v for k, v in self.bilin_polys.items() if k[0] == k[1]]),
            key=str
        )
        self.bilin_poly_list = sorted(
            utils.unique_elements(self.bilin_polys.values()),
            key=str
        )

    def are_homotopy_equivalent(self, aug_1, aug_2):
        """The augmentations are homotopy equivalent iff the induced map on bilinearized homology to the base field
        given by aug = (aug_1 - aug_2) is zero. How do we figure this out using only generators of the underlying vector
        spaces of generators? Since aug(del(x)) always vanishes, it is already zero on Image(del). Therefore aug == 0
        on homology iff it vanishes on ker del. Another way to check is if the lin homologies agree and if the bilin
        homologies agree with those of the lin homologies.

        This is actually annoying to do in sympy and easy to do in sage :(
        However, we can do a trick... Represent ker as the intersection of the varieties determined by the row vectors
        of a marix representation of del. Say these linear equations are p_i. Since aug != 0 iff 1 in Im(aug), we see
        that aug restricted to ker is non-zero iff there is a solution to [p_1, ..., p_k, aug + 1]. This can be worked
        out with the equation solver using existence_only=True.

        It would actually save time also to use transitivity to avoid checking all of the homotopy equiv classes.

        :param aug_1:
        :param aug_2:
        :return:
        """
        raise NotImplementedError()

    @utils.log_start_stop
    def set_aug_data(self):
        """Set variables needed to compute augmentations.

        :return: None
        """
        zero_graded_symbols = [g for g in self.symbols if self.gradings[g] == 0]
        LOG.info(f"DGA has {len(zero_graded_symbols)} generators with 0 grading")
        self.aug_symbols_to_comm_symbols = {g: sympy.Symbol(str(g), commutative=True) for g in zero_graded_symbols}
        self.aug_comm_symbols_to_symbols = {v: k for k, v in self.aug_symbols_to_comm_symbols.items()}
        # Only need to mod out by differentials of degree 1 elements
        d_expressions = {k: v.expression for k, v in self.differentials.items() if self.gradings[k] == 1}
        LOG.info(f"Differentials required to compute augs: {len(d_expressions)}")
        # Set all non-zero graded elements to zero.
        zero_substitutions = {g: 0 for g in self.symbols if self.gradings[g] != 0}
        # Make polynomials in commutative variables of the deg=0 generators.
        # This is required to apply zero_set which only works with commuting variables!
        LOG.info(f"Eliminating degree-non-zero terms from polys")
        aug_polys = {
            k: sympy.sympify(v).subs(zero_substitutions)
            for k, v in d_expressions.items()
        }
        LOG.info(f"Replacing non-commutative symbols with commutative")
        aug_polys = {
            k: v.subs(self.aug_symbols_to_comm_symbols)
            for k, v in aug_polys.items()
        }
        aug_polys = utils.unique_elements(aug_polys.values())
        if 0 in aug_polys:
            aug_polys.remove(0)
        LOG.info(f"Polynomials required to compute augs after simplification: {len(aug_polys)}")
        self.aug_polys = aug_polys
        # Create table to help inform batch size
        # TODO: This is replicated in batch_zero_set. Should make into a function
        poly_to_symbols = {p: polynomials.poly_symbols(p) for p in aug_polys}
        symbol_freq = Counter()
        for symbol_set in poly_to_symbols.values():
            symbol_freq.update(list(symbol_set))
        symbol_freq = dict(symbol_freq)
        for sym in self.aug_comm_symbols_to_symbols.keys():
            if sym not in symbol_freq.keys():
                symbol_freq[sym] = 0
        symbol_table = [
            {
                "symbol": k,
                "freq": v
            }
            for k, v in symbol_freq.items()
        ]
        symbol_table.sort(key=lambda d: d["freq"], reverse=True)
        running_symbol_set = set()
        for i in range(len(symbol_table)):
            running_symbol_set.add(symbol_table[i]["symbol"])
            running_polys = [k for k, v in poly_to_symbols.items() if v.issubset(running_symbol_set)]
            symbol_table[i]["n_cum_polys"] = len(running_polys)
            symbol_table[i]["poly_to_sym_ratio"] = 1.0*len(running_polys) / (i+1)
        self.aug_analysis_table = symbol_table
        LOG.info(f"Aug analysis table: \n" + "\n".join([str(r) for r in symbol_table]))
        self._aug_vars_set = True

    @utils.log_start_stop
    def set_augmentations(self, batch_size=None, filtered=False, decompress=True):
        # this pattern is bad, because we allow 0 coeff_mod upon instance creation and then this method always runs
        if self.coeff_mod == 0:
            raise ValueError("We cannot search for augmentations over ZZ. It's too difficult :(")
        if filtered and batch_size is not None:
            raise ValueError("Cannot do batch and filtered simultaneously")
        if not self._aug_vars_set:
            self.set_aug_data()
        zero_graded_symbols = list(self.aug_symbols_to_comm_symbols.keys())
        comm_symbols = list(self.aug_symbols_to_comm_symbols.values())
        if batch_size is not None:
            comm_augmentations = polynomials.batch_zero_set(
                polys=self.aug_polys, symbols=comm_symbols, modulus=self.coeff_mod, batch_size=batch_size)
        elif filtered:
            comm_filtration_levels = {
                v: self.filtration_levels[k]
                for k, v in self.aug_symbols_to_comm_symbols.items()
            }
            comm_augmentations = polynomials.filtered_zero_set(
                polys=self.aug_polys, symbols=comm_symbols,
                filtration_levels=comm_filtration_levels, modulus=self.coeff_mod)
        else:
            comm_augmentations = polynomials.zero_set(
                polys=self.aug_polys, symbols=comm_symbols, modulus=self.coeff_mod)
        self.augmentations_compressed = comm_augmentations
        self.n_augs_compressed = len(self.augmentations_compressed)
        LOG.info(f"Found {self.n_augs_compressed} compressed augmentations of DGA")
        if decompress:
            self.decompress_augmentations()
        else:
            self._set_n_augs_from_compressed()
            LOG.info(f"Found {self.n_augs} (uncompressed) augmentations of DGA")

    def get_decompressed_augmentations(self, start_i=None, end_i=None, fill_na=None):
        if fill_na is None:
            fill_na = self.aug_fill_na
        if fill_na is not None:
            LOG.info(f"Working with a subset of augmentations using default value {fill_na}")
        if start_i is None:
            start_i = 0
        if end_i is None:
            end_i = len(self.augmentations_compressed)
        augs_compressed = self.augmentations_compressed[start_i:end_i]
        augmentations = []
        comm_augmentations = polynomials.expand_zero_set_from_unset(
            augs_compressed, modulus=self.coeff_mod, fill_na=fill_na
        )
        for aug in comm_augmentations:
            augmentations.append({self.aug_comm_symbols_to_symbols[k]: v for k, v in aug.items()})
        return augmentations

    @utils.log_start_stop
    def decompress_augmentations(self, fill_na=None):
        # comm_augs will be a list of dicts whose keys are the commutative symbols.
        # We need to switch them back!
        if fill_na is None:
            fill_na = self.aug_fill_na
        self.augmentations = self.get_decompressed_augmentations(fill_na=fill_na)
        self.n_augs = len(self.augmentations)
        LOG.info(f"Found {self.n_augs} augmentations of DGA")

    def _set_n_augs_from_compressed(self):
        count = 0
        for aug_c in self.augmentations_compressed:
            n_unset_vars = len([k for k, v in aug_c.items() if v is polynomials.UNSET_VAR])
            count += self.coeff_mod ** n_unset_vars
        self.n_augs = count
