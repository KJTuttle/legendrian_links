from collections import Counter
from math import prod
import sympy
import algebra
import utils


LOG = utils.get_logger(__name__)
ZZ = sympy.ZZ
ZZ2 = sympy.GF(2)
HALF = sympy.Rational(1, 2)


def graded_by_valid_or_except(graded_by):
    if not (graded_by.is_ZZ or (graded_by.is_FiniteField and graded_by.mod == 2)):
        raise ValueError(f"Generator must be graded by Z or Z2. Graded by {str(graded_by)}")


class Chord(object):
    """Object which carries crossing information.
    Should be able to see the line segments at the positive and negative ends of a chord.
    From this we should be able to implement signs for holomorphic disks."""


    def __init__(self, top_line_segment, bottom_line_segment):
        self.top_line_segment = top_line_segment
        self.bottom_line_segment = bottom_line_segment
        self._set_xy()
        self._set_sign()

    def __repr__(self):
        return str(self.x)

    def is_composable_with(self, chord):
        return self.top_line_segment.knot_label == chord.bottom_line_segment.knot_label

    def is_pure(self):
        return self.is_composable_with(self)

    def _set_xy(self):
        self.x = self.top_line_segment.x_left
        self.y = self.top_line_segment.y_left

    def _set_sign(self):
        if self.top_line_segment.orientation is None or self.bottom_line_segment.orientation is None:
            raise ValueError(f"Trying to initialize crossing with unoriented line segments. "
                             f"Top {self.top_line_segment.to_array()}, "
                             f"bottom {self.bottom_line_segment.to_array()}")
        self.sign = 1 if self.top_line_segment.orientation == self.bottom_line_segment.orientation else -1


class LCHGenerator(object):

    def __init__(self, chord, graded_by, capping_path=None):
        self.chord = chord
        self.capping_path = capping_path
        graded_by_valid_or_except(graded_by)
        self.graded_by = graded_by
        self._set_grading()
        self._set_x()
        self._set_symbol()

    def __repr__(self):
        return "{" + str(self.chord.x) + "}"

    def _set_grading(self):
        if self.graded_by.is_ZZ:
            maslov = self.capping_path.rotation_number - HALF
            self.grading = maslov
        else:
            self.grading = 0 if self.chord.sign == 1 else 1

    def _set_x(self):
        self.x = self.chord.x

    def _set_symbol(self):
        self.symbol = sympy.Symbol(str(self), commutative=False)


class RSFTGenerator(object):

    def __init__(self, word, graded_by, capping_paths=None):
        for chord in word:
            if not isinstance(chord, Chord):
                raise ValueError("Trying to create WordOfChord object not from a list of chords")
        self.word = word
        self.length = len(word)
        graded_by_valid_or_except(graded_by)
        self.graded_by = graded_by
        self.capping_paths = capping_paths
        self._validate_capping_paths()
        self._set_grading()
        self._set_knot_labels()
        self._set_x()
        self._set_symbol()

    def __repr__(self):
         return "{" + ",".join([str(chord.x) for chord in self.word]) + "}"

    def _validate_capping_paths(self):
        if self.graded_by.is_ZZ:
            if self.capping_paths is None:
                raise ValueError(f"Z graded RSFT generators need capping paths.")
            if len(self.capping_paths) != len(self.word):
                raise ValueError(f"len(capping_paths) = {len(self.capping_paths)} != len(word) = {len(self.word)}")

    def _set_grading(self):
        if self.graded_by.is_ZZ:
            self.grading = -1 + HALF*len(self.word) + sum([path.rotation_number for path in self.capping_paths])
        else:
            self.grading = (-1 + len(self.word) + len([chord for chord in self.word if chord.sign == -1])) % 2

    def _set_knot_labels(self):
        knot_labels = []
        for chord in self.word:
            kl = chord.top_line_segment.knot_label
            if kl is None:
                raise RuntimeError(f"Trying to set knot labels for {str(self)} with unlabeled chord {chord}")
            knot_labels.append(kl)
        self.knot_labels = knot_labels

    def _set_x(self):
        self.x = [chord.x for chord in self.word]

    def _set_symbol(self):
        self.symbol = sympy.Symbol(str(self), commutative=False)


class DiskCorner(object):
    """Convex corner of a disk in the Lagrangian projection. Corner type tells the direction it points."""
    CORNER_TYPES = {
        'l': '+',
        'r': '+',
        'u': '-',
        'd': '-'
    }

    def __init__(self, chord, corner):
        self.chord = chord
        if corner not in self.CORNER_TYPES.keys():
            raise ValueError(f"Incoming type {corner} not in {self.CORNER_TYPES}")
        self.corner = corner
        self.pos_neg = self.CORNER_TYPES[self.corner]

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {
            'x': self.chord.top_line_segment.x_left,
            'from_knot': self.chord.bottom_line_segment.knot_label,
            'to_knot': self.chord.top_line_segment.knot_label,
            'corner': self.corner,
            'pos_neg': self.pos_neg
        }


class DiskSegment(object):

    def __init__(self, x, disk_corner, top_left_ls, bottom_left_ls, top_right_ls, bottom_right_ls):
        self.x = x
        self.disk_corner = disk_corner
        self.top_left_ls = top_left_ls
        self.bottom_left_ls = bottom_left_ls
        self.top_right_ls = top_right_ls
        self.bottom_right_ls = bottom_right_ls
        self.corner = None
        if disk_corner is not None:
            self.corner = disk_corner.corner
        self._set_y_values()

    def _set_y_values(self):
        self.left_y_values = [
            getattr(self.top_left_ls, 'y_left', None),
            getattr(self.bottom_left_ls, 'y_left', None)
        ]
        self.right_y_values = [
            getattr(self.top_right_ls, 'y_right', None),
            getattr(self.bottom_right_ls, 'y_right', None)
        ]

    def to_dict(self):
        output = dict()
        if self.disk_corner is not None:
            output = self.disk_corner.to_dict()
        output['x'] = self.x
        output['left_endpoints'] = self.left_y_values
        output['right_endpoints'] = self.right_y_values
        return output


class DiskSegmentGraph(object):
    """A directed graph which encodes how DiskSegments with x coordinates are linked together"""

    def __init__(self, n_segments):
        self.vertices = list()
        self.edges = list()
        self.n_segments = n_segments

    def add_vertex(self, disk_segment):
        self.vertices.append(disk_segment)

    def vertex_is_initial(self, i):
        vertex = self.vertices[i]
        if (vertex.x == 0) or (vertex.corner == 'l'):
            return True
        return False

    def vertex_is_terminal(self, i):
        vertex = self.vertices[i]
        if (vertex.x == self.n_segments - 1) or (vertex.corner == 'r'):
            return True
        return False

    def vertex_dead_end(self, i):
        for e in self.edges:
            if e[0] == i:
                return True
        return False

    def vertices_are_consecutive(self, i, j):
        v_left = self.vertices[i]
        v_right = self.vertices[j]
        if v_right.x != v_left.x + 1:
            return False
        return v_left.right_y_values == v_right.left_y_values

    def compute_edges(self):
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices)):
                if self.vertices_are_consecutive(i, j):
                    self.edges.append([i, j])

    def compute_paths_from_vertex(self, i):
        if self.vertex_is_terminal(i):
            return [[i]]
        outgoing_vertices = [e[1] for e in self.edges if e[0] == i]
        return [[i] + p for v in outgoing_vertices for p in self.compute_paths_from_vertex(v)]

    def compute_paths(self):
        # TODO: Divide and conquer is suboptimal. Can instead use dynamic algo which is faster.
        # This may actually be important for very large numbers of disks.
        paths = []
        for i in range(len(self.vertices)):
            if self.vertex_is_initial(i):
                paths += self.compute_paths_from_vertex(i)
        paths = [p for p in paths if self.vertex_is_terminal(p[-1])]
        return paths

    def path_to_disk(self, index_path):
        return Disk([self.vertices[i] for i in index_path])

    def compute_disks(self):
        paths = self.compute_paths()
        return [self.path_to_disk(p) for p in paths]


class Disk(object):
    """A disk is a list of disk segments such that the right endpoints of one segment
    agree with the left endpoints of the next. This is supposed to model an index 1
    J-disk in the Lagrangian projection determined by a Lagrangian resolution."""

    def __init__(self, disk_segments):
        self.disk_segments = disk_segments
        self._set_disk_corners()

    def __repr__(self):
        return str([str(dc) for dc in self.disk_corners])

    def get_endpoints(self):
        return [ds.get_endpoint_y_values() for ds in self.disk_segments]

    def _set_disk_corners(self):
        disk_corners = []
        disk_segments = sorted(self.disk_segments, key=lambda disk_seg: disk_seg.x)
        # To go CCW, left and down corners go left-to-right. Vice versa for up and right corners
        for ds in disk_segments:
            if ds.disk_corner is not None:
                if ds.disk_corner.corner in ['l', 'd']:
                    disk_corners.append(ds.disk_corner)
        for ds in reversed(disk_segments):
            if ds.disk_corner is not None:
                if ds.disk_corner.corner in ['u', 'r']:
                    disk_corners.append(ds.disk_corner)
        self.disk_corners = disk_corners
        self.pos_corners = [c for c in self.disk_corners if c.pos_neg == '+']
        self.neg_corners = [c for c in self.disk_corners if c.pos_neg == '-']

    def is_lch(self):
        return len(self.pos_corners) == 1


class PlatSegment(object):

    def __init__(self, line_segments, chord=None, left_close=False, right_close=False):
        self.line_segments = line_segments
        self.chord = chord
        self.n_strands = len(line_segments)
        self.left_close = left_close
        self.right_close = right_close
        self._set_xy()

    def to_dict(self):
        return {
            "x": self.x,
            "n_strands": self.n_strands,
            "left_close": self.left_close,
            "right_close": self.right_close,
            "crossing_y": self.crossing_y
        }

    def get_disk_segments(self):
        # enumerate all of the disk segments and return them as a set
        disk_segments = []
        # if it is a left close of a right close
        if self.left_close:
            line_segments = sorted(self.line_segments, key=lambda ls: ls.y_right)
            for i in range(len(line_segments)):
                if i % 2 == 0:
                    disk_segments.append(
                        DiskSegment(
                            x=self.x, disk_corner=None,
                            top_left_ls=line_segments[i], top_right_ls=line_segments[i],
                            bottom_left_ls=line_segments[i+1], bottom_right_ls=line_segments[i+1]
                        )
                    )
            return disk_segments
        line_segments = sorted(self.line_segments, key=lambda ls: ls.y_left)
        if self.right_close:
            for i in range(len(line_segments)):
                if i % 2 == 0:
                    disk_segments.append(
                        DiskSegment(
                            x=self.x, disk_corner=None,
                            top_left_ls=line_segments[i], top_right_ls=line_segments[i],
                            bottom_left_ls=line_segments[i + 1], bottom_right_ls=line_segments[i + 1]
                        )
                    )
            return disk_segments

        # disk segments without crossings
        for top_ls in line_segments:
            bottom_line_segments = [
                ls for ls in line_segments if ls.y_left > top_ls.y_left and ls.y_right > top_ls.y_right
            ]
            for bottom_ls in bottom_line_segments:
                disk_segments.append(
                    DiskSegment(
                        x=self.x,
                        disk_corner=None,
                        top_left_ls=top_ls, top_right_ls=top_ls,
                        bottom_left_ls=bottom_ls, bottom_right_ls=bottom_ls
                    )
                )

        # a crossing appears in all other cases
        chord = self.chord
        # with a crossing on the bottom
        bottom_left_ls = chord.top_line_segment
        bottom_right_ls = chord.bottom_line_segment
        top_line_segments = [
            ls for ls in line_segments if ls.y_left < bottom_left_ls.y_left and ls.y_right < bottom_right_ls.y_right
        ]
        for ls in top_line_segments:
            disk_segments.append(DiskSegment(
                x=self.x,
                disk_corner=DiskCorner(chord=chord, corner='d'),
                top_right_ls=ls, top_left_ls=ls,
                bottom_left_ls=bottom_left_ls, bottom_right_ls=bottom_right_ls
            ))
        # with a crossing on the top
        top_right_ls = chord.top_line_segment
        top_left_ls = chord.bottom_line_segment
        bottom_line_segments = [
            ls for ls in line_segments if ls.y_left > bottom_left_ls.y_left and ls.y_right > bottom_right_ls.y_right
        ]
        for ls in bottom_line_segments:
            disk_segments.append(
                DiskSegment(
                    x=self.x,
                    disk_corner=DiskCorner(chord=chord, corner='u'),
                    top_right_ls=top_right_ls, top_left_ls=top_left_ls,
                    bottom_left_ls=ls, bottom_right_ls=ls
                )
            )
        # with a crossing on the left
        disk_segments.append(
            DiskSegment(
                x=self.x,
                disk_corner=DiskCorner(chord=chord, corner='l'),
                top_left_ls=None, bottom_left_ls=None,
                top_right_ls=chord.bottom_line_segment, bottom_right_ls=chord.top_line_segment
            )
        )
        # with a crossing on the right
        disk_segments.append(
            DiskSegment(
                x=self.x,
                disk_corner=DiskCorner(chord=chord, corner='r'),
                top_left_ls=chord.top_line_segment, bottom_left_ls=chord.bottom_line_segment,
                top_right_ls=None, bottom_right_ls=None
            )
        )
        return disk_segments

    def _set_xy(self):
        x_left_values = list(set([ls.x_left for ls in self.line_segments]))
        n_x_left_values = len(x_left_values)
        if n_x_left_values > 1:
            raise ValueError(f"Found {n_x_left_values} different left x values for line segments in PlatSegment")
        self.x = x_left_values[0]
        self.crossing_y = None
        if self.chord is not None:
            self.crossing_y = self.chord.top_line_segment.y_left


class LineSegment(object):
    ORIENTATIONS = ['l', 'r']

    def __init__(self, x_left, y_left, x_right, y_right):
        self.x_left = x_left
        self.y_left = y_left
        self.x_right = x_right
        self.y_right = y_right
        self.orientation = None
        self.knot_label = None
        self.t_label = False

    def to_array(self):
        return [[self.x_left, self.y_left], [self.x_right, self.y_right]]

    def __repr__(self):
        return str(self.to_array())

    def set_orientation(self, o):
        if o not in self.ORIENTATIONS:
            raise ValueError(f'LineSegment orientation must be in {self.ORIENTATIONS}')
        self.orientation = o

    def toggle_t_label(self):
        self.t_label = not self.t_label

    def set_knot_label(self, kl):
        self.knot_label = kl


class CappingPath(object):
    """Records capping path starting at the top endpoint of start_chord and ending at the initial point of end_chord.
    CappingPath's always follow orientation of the knot in which they're contained.
    Rotation numbers are computed assuming that all .
    """

    def __init__(self, start_chord, end_chord, line_segments, t_label):
        self.start_chord = start_chord
        self.end_chord = end_chord
        self.line_segments = line_segments
        self.t_label = t_label
        self._set_touches_basepoint()
        self._set_rotation_number()

    def __repr__(self):
        return ", ".join([str(ls) for ls in self.line_segments])

    def _set_touches_basepoint(self):
        self.touches_basepoint = any([segment.t_label for segment in self.line_segments])

    def _set_rotation_number(self):
        """Rotation numbers will be multiples of one half, given by rotation angles divided by pi.
        Stored as sympy.Rational to avoid float issues"""
        switches = [
            (self.line_segments[i], self.line_segments[i+1])
            for i in range(len(self.line_segments) - 1)
            if self.line_segments[i].orientation != self.line_segments[i+1].orientation
        ]

        def sign_of_switch(switch):
            if switch[0].orientation == 'r':
                return 1 if switch[0].y_left > switch[1].y_left else -1
            else:
                return -1 if switch[0].y_right > switch[1].y_right else 1

        rotation_number = sum([sign_of_switch(switch) for switch in switches])
        # we get a quarter at both the initial and terminal points
        rotation_number += HALF

        self.rotation_number = rotation_number


class PlatDiagram(object):

    def __init__(self, n_strands, front_crossings=[], auto_lch=False, auto_rsft=False):
        self.n_strands = n_strands
        self.front_crossings = front_crossings

        self._set_line_segments()
        self.max_x_left = max([ls.x_left for ls in self.line_segments])
        self._label_line_segments()
        # wait to set crossings until the line segments have been labeled
        self._set_chords()
        self._set_knots()
        self._set_composable_pairs()
        self._set_capping_paths()
        self._set_plat_segments()
        self._set_disk_graph()
        self._set_disks()
        self._set_disk_corners()
        self.auto_lch = auto_lch
        if auto_lch:
            self.set_lch()
        self.auto_rsft = auto_rsft
        if auto_rsft:
            self.set_rsft()

    def get_line_segment_array(self):
        return [
            {
                'array': ls.to_array(),
                'knot_label': ls.knot_label,
                'orientation': ls.orientation,
                't_label': ls.t_label
            }
            for ls in self.line_segments]

    def get_line_segment_by_right_xy(self, x, y):
        search_results = [ls for ls in self.line_segments if ls.x_right == x and ls.y_right == y]
        n_results = len(search_results)
        if n_results != 1:
            raise RuntimeError(
                f'Found {n_results} in search for line segment by right_xy, rather than 1 at x,y={x},{y}. '
                f'Line segments available: {[ls.to_array() for ls in self.line_segments]}')
        return search_results[0]

    def get_line_segment_by_left_xy(self, x, y):
        search_results = [ls for ls in self.line_segments if ls.x_left == x and ls.y_left == y]
        n_results = len(search_results)
        if n_results != 1:
            raise RuntimeError(
                f'Found {n_results} in search for line segment by left_xy, rather than 1 at x,y={x},{y}. '
                f'Line segments available: {[ls.to_array() for ls in self.line_segments]}')
        return search_results[0]

    def get_next_line_segment(self, line_segment, reverse=False):
        orientation = line_segment.orientation
        if reverse:
            orientation = 'r' if orientation == 'l' else 'r'
        if orientation == 'r':
            if line_segment.x_left == self.max_x_left:
                if line_segment.y_left > line_segment.y_right:
                    return self.get_line_segment_by_left_xy(x=line_segment.x_left, y=line_segment.y_left - 1)
                else:
                    return self.get_line_segment_by_left_xy(x=line_segment.x_left, y=line_segment.y_left + 1)
            else:
                return self.get_line_segment_by_left_xy(x=line_segment.x_right, y=line_segment.y_right)
        else:
            if line_segment.x_left == 0:
                if line_segment.y_left > line_segment.y_right:
                    return self.get_line_segment_by_right_xy(x=line_segment.x_right, y=line_segment.y_right + 1)
                else:
                    return self.get_line_segment_by_right_xy(x=line_segment.x_right, y=line_segment.y_right - 1)
            else:
                return self.get_line_segment_by_right_xy(x=line_segment.x_left, y=line_segment.y_left)

    def line_segment_is_incoming_to_left_up_cusp(self, line_segment):
        if not (line_segment.x_left == 0 and line_segment.orientation == 'l'):
            return False
        next_ls = self.get_next_line_segment(line_segment)
        return next_ls.y_right < line_segment.y_right

    def line_segment_is_incoming_to_left_down_cusp(self, line_segment):
        if not (line_segment.x_left == 0 and line_segment.orientation == 'l'):
            return False
        next_ls = self.get_next_line_segment(line_segment)
        return next_ls.y_right > line_segment.y_right

    def line_segment_is_incoming_to_right_up_cusp(self, line_segment):
        if not (line_segment.x_left == self.max_x_left and line_segment.orientation == 'r'):
            return False
        next_ls = self.get_next_line_segment(line_segment)
        return next_ls.y_left < line_segment.y_left

    def line_segment_is_incoming_to_right_down_cusp(self, line_segment):
        if not (line_segment.x_left == self.max_x_left and line_segment.orientation == 'r'):
            return False
        next_ls = self.get_next_line_segment(line_segment)
        return next_ls.y_left > line_segment.y_left

    def get_capping_path(self, start_chord, end_chord):
        search_results = [p for p in self.capping_paths if p.start_chord == start_chord and p.end_chord == end_chord]
        n_results = len(search_results)
        if n_results != 1:
            raise RuntimeError(f"Found {n_results} possible capping paths for"
                               f"start_chord={start_chord} and"
                               f"end_chord={end_chord}")
        return search_results[0]

    def get_chord_from_x(self, x):
        search_results = [c for c in self.chords if c.x == x]
        if len(search_results) != 1:
            raise ValueError(f"{len(search_results)} possible chords found with x={x}")
        return search_results[0]

    def get_lch_generator_from_chord(self, chord):
        search_results = [g for g in self.lch_generators if g.chord == chord]
        n_results = len(search_results)
        if n_results != 1:
            raise RuntimeError(f"Found {n_results} possible LCH generators for chord {chord}")
        return search_results[0]

    def get_rsft_generator_from_word(self, word, raise_if_not_found=True):
        search_results = [w for w in self.rsft_generators if w.word == word]
        n_results = len(search_results)
        if n_results > 1:
            raise RuntimeError(f"Found {n_results} possible RSFT generators for word {word}")
        if n_results == 0:
            if raise_if_not_found:
                raise ValueError(f"No RSFT generator found matching word {word}")
            return
        return search_results[0]

    def word_is_admissible(self, word, cyclic=True):
        """Returns True / False for if the word of chords is `admissible`. This means that each pair of chords is
        composable and that the endpoints of chords hit eat knot at most once.

        :param word: list of chords
        :param cyclic: Are we requiring that word[-1] is composable with word[0]?
        :return: Boolean
        """
        if len(word) == 1:
            if cyclic:
                return (word[0], word[0]) in self.composable_pairs
            else:
                return True
        else:
            output = True
            output &= all([(word[i], word[i+1]) in self.composable_pairs for i in range(len(word) - 1)])
            endpoint_labels = [c.top_line_segment.knot_label for c in word]
            output &= utils.is_nonrepeating(endpoint_labels)
            if not cyclic:
                return output
            output &= (word[-1], word[0]) in self.composable_pairs
            return output

    def set_lch(self):
        self._set_lch_graded_by()
        self._set_lch_generators()
        self._set_lch_dga()

    def set_rsft(self):
        self._set_composable_admissible_words()
        self._set_rsft_graded_by()
        self._set_rsft_generators()
        self._set_rsft_dga()

    def _set_line_segments(self):
        lines = []
        lag_crossings_from_right_cusps = [i for i in range(self.n_strands) if i % 2 == 0]
        lag_crossings = self.front_crossings + lag_crossings_from_right_cusps

        # make the left closing part
        x = 0
        for i in range(self.n_strands):
            if i % 2 == 0:
                lines.append(LineSegment(x_left=0, y_left=i + .5, x_right=1, y_right=i))
            else:
                lines.append(LineSegment(x_left=0, y_left=i - .5, x_right=1, y_right=i))
        # make the segments at x values where there are crossings
        if lag_crossings is not None:
            for fcy in lag_crossings:
                x += 1
                for s in range(self.n_strands):
                    if s == fcy:
                        lines.append(LineSegment(x_left=x, y_left=s, x_right=x + 1, y_right=s + 1))
                    elif s == fcy + 1:
                        lines.append(LineSegment(x_left=x, y_left=s, x_right=x + 1, y_right=s - 1))
                    else:
                        lines.append(LineSegment(x_left=x, y_left=s, x_right=x + 1, y_right=s))
        # make the right closing part
        x += 1
        for i in range(self.n_strands):
            if i % 2 == 0:
                lines.append(LineSegment(x_left=x, y_left=i, x_right=x + 1, y_right=i + .5))
            else:
                lines.append(LineSegment(x_left=x, y_left=i, x_right=x + 1, y_right=i - .5))

        self.line_segments = lines

    def _label_line_segments(self):
        knot_label = 0
        # The while loops are in danger of being infinite. Cut at an n to debug.
        n = 0
        while True:
            unlabeled_line_segments = [ls for ls in self.line_segments if ls.knot_label is None]
            n_unlabeled_line_segments = len(unlabeled_line_segments)
            if n_unlabeled_line_segments == 0:
                break
            initial_ls = unlabeled_line_segments[0]
            initial_ls.set_knot_label(knot_label)
            initial_ls.set_orientation('r')
            ls = self._label_next_line_segment(initial_ls)
            while ls != initial_ls:
                ls = self._label_next_line_segment(ls)
                n += 1
            knot_label += 1
        self.n_components = len(set([ls.knot_label for ls in self.line_segments]))
        # add a t label to exactly one line segment in each component of the link
        for knot_label in range(self.n_components):
            knot = [ls for ls in self.line_segments if ls.knot_label == knot_label]
            knot[0].toggle_t_label()

    def _label_next_line_segment(self, ls):
        if ls not in self.line_segments:
            raise ValueError('line_segment does not belong to PlatDiagram instance')
        if ls.orientation is None:
            raise ValueError('Cannot find next line segment if it is not orientated')
        if ls.knot_label is None:
            raise ValueError('Cannot find knot_label for line_segment')

        max_x_right = self.max_x_left + 1
        if ls.orientation == 'r':
            if ls.x_right < max_x_right:
                next_ls = self.get_line_segment_by_left_xy(x=ls.x_right, y=ls.y_right)
                next_ls.set_orientation('r')
            else:
                if ls.y_right < ls.y_left:
                    next_ls = self.get_line_segment_by_left_xy(x=ls.x_left, y=ls.y_left - 1)
                else:
                    next_ls = self.get_line_segment_by_left_xy(x=ls.x_left, y=ls.y_left + 1)
                next_ls.set_orientation('l')
        else:
            if ls.x_left > 0:
                next_ls = self.get_line_segment_by_right_xy(x=ls.x_left, y=ls.y_left)
                next_ls.set_orientation('l')
            else:
                if ls.y_right < ls.y_left:
                    next_ls = self.get_line_segment_by_right_xy(x=ls.x_right, y=ls.y_right + 1)
                else:
                    next_ls = self.get_line_segment_by_right_xy(x=ls.x_right, y=ls.y_right - 1)
                next_ls.set_orientation('r')

        next_ls.set_knot_label(ls.knot_label)
        return next_ls

    def _set_chords(self):
        chords = []
        for x in range(0, self.max_x_left):
            x_segments = [ls for ls in self.line_segments if ls.x_left == x]
            for y in range(self.n_strands - 1):
                if x == 0:
                    top_left_ls = [ls for ls in x_segments if ls.y_right == y][0]
                    bottom_left_ls = [ls for ls in x_segments if ls.y_right == y + 1][0]
                else:
                    top_left_ls = [ls for ls in x_segments if ls.y_left == y][0]
                    bottom_left_ls = [ls for ls in x_segments if ls.y_left == y + 1][0]
                are_crossing = (top_left_ls.y_right == y + 1) and (bottom_left_ls.y_right == y)
                if are_crossing:
                    chords.append(Chord(top_line_segment=top_left_ls, bottom_line_segment=bottom_left_ls))
        self.chords = chords

    def _set_knots(self):
        knot_labels = sorted(list(set([ls.knot_label for ls in self.line_segments])))
        knots = list()
        for kl in knot_labels:
            knot_line_segments = [ls for ls in self.line_segments if ls.knot_label == kl]
            # compute tb as writhe in the Lagrangian projection
            self_chords = [c for c in self.chords
                           if c.top_line_segment.knot_label == kl and c.bottom_line_segment.knot_label == kl]
            tb = sum([c.sign for c in self_chords])
            # compute rot from the left and right cusps
            rot = 0
            for ls in knot_line_segments:
                if self.line_segment_is_incoming_to_left_down_cusp(ls):
                    rot += 1
                if self.line_segment_is_incoming_to_right_up_cusp(ls):
                    rot += 1
                if self.line_segment_is_incoming_to_left_up_cusp(ls):
                    rot -= 1
                if self.line_segment_is_incoming_to_right_down_cusp(ls):
                    rot -= 1
            rot = rot // 2

            knots.append({
                'line_segments': knot_line_segments,
                'tb': tb,
                'rot': rot
            })

        self.knots = knots
        self.link_is_connected = len(self.knots) == 1

    def _set_lch_graded_by(self):
        """We only use Z gradings for LCH when our link is a not with zero rotation. This can be improved."""
        if len(self.knots) == 1 and self.knots[0]['rot'] == 0:
            self.lch_graded_by = ZZ
        else:
            self.lch_graded_by = ZZ2

    def _set_rsft_graded_by(self):
        """Only use Z gradings if all knot components have rot=0. This can be improved."""
        if all([k['rot'] == 0 for k in self.knots]):
            self.rsft_graded_by = ZZ
        else:
            self.rsft_graded_by = ZZ2

    def _set_composable_pairs(self):
        self.composable_pairs = [
            (chord_1, chord_2) for chord_1 in self.chords for chord_2 in self.chords
            if chord_1.is_composable_with(chord_2)
        ]

    def _set_capping_paths(self):
        self.capping_paths = []
        for start_chord, end_chord in self.composable_pairs:
            start_segment = start_chord.top_line_segment
            end_segment = end_chord.bottom_line_segment
            capping_path_segments = [start_segment]
            path_is_t_labeled = False
            while capping_path_segments[-1] != end_segment:
                next_segment = self.get_next_line_segment(capping_path_segments[-1])
                if next_segment.t_label:
                    path_is_t_labeled = True
                capping_path_segments.append(next_segment)
            self.capping_paths.append(
                CappingPath(
                    start_chord=start_chord,
                    end_chord=end_chord,
                    line_segments=capping_path_segments,
                    t_label=path_is_t_labeled
                )
            )

    def _set_plat_segments(self):
        max_left_x = max([ls.x_left for ls in self.line_segments])
        plat_segments = []
        for x in range(0, max_left_x + 1):
            left_x_segments = [ls for ls in self.line_segments if ls.x_left == x]
            left_close = x == 0
            right_close = x == max_left_x
            chords = [c for c in self.chords
                      if c.bottom_line_segment in left_x_segments or c.top_line_segment in left_x_segments]
            if len(chords) > 1:
                raise ValueError(f'More than 1 chord at a plat segment')
            chord = chords[0] if len(chords) == 1 else None
            ps = PlatSegment(left_x_segments, chord = chord, left_close=left_close, right_close=right_close)
            plat_segments.append(ps)
        self.plat_segments = plat_segments

    def _set_disk_graph(self):
        disk_graph = DiskSegmentGraph(n_segments=len(self.plat_segments))
        for x in range(len(self.plat_segments)):
            segment = self.plat_segments[x]
            disk_segments = segment.get_disk_segments()
            for d in disk_segments:
                disk_graph.add_vertex(d)
        disk_graph.compute_edges()
        self.disk_graph = disk_graph

    def _set_disks(self):
        self.disks = self.disk_graph.compute_disks()

    def _set_disk_corners(self):
        self.disk_corners = [d.disk_corners for d in self.disks]

    def _set_lch_generators(self):
        lch_generators = []
        for chord in self.chords:
            capping_path = (
                self.get_capping_path(start_chord=chord, end_chord=chord)
                if self.lch_graded_by == ZZ else None
            )
            lch_generators.append(
                LCHGenerator(
                    chord=chord,
                    graded_by=self.lch_graded_by,
                    capping_path=capping_path)
            )
        self.lch_generators = lch_generators

    def _set_lch_dga(self):
        lch_disks = [d for d in self.disks if d.is_lch()]
        lch_del = {g: 0 for g in self.lch_generators}
        for d in lch_disks:
            disk_corners = d.disk_corners.copy()
            # cyclically rotate disk until positive corner is in the last (-1) position
            while disk_corners[-1].pos_neg == '-':
                disk_corners = utils.rotate(disk_corners, 1)
            pos_chord = disk_corners.pop().chord
            pos_generator = self.get_lch_generator_from_chord(pos_chord)
            if len(disk_corners) > 0:
                neg_word = [self.get_lch_generator_from_chord(dc.chord).symbol for dc in disk_corners]
            else:
                neg_word = [1]
            # Add negative word of chords as a summand to the LCH
            # TODO: Add sign contributions
            lch_del[pos_generator] += prod(neg_word)
        lch_del = {
            g.symbol: algebra.Differential(lch_del[g])
            for g in lch_del.keys()
        }
        gradings = {g.symbol: g.grading for g in self.lch_generators}
        self.lch_dga = algebra.DGA(gradings=gradings, differentials=lch_del, coeff_mod=2)

    def _set_composable_admissible_words(self):
        """Store all composable admissible (not necessarily cyclic) words in memory for lookup access"""
        # length 1 composable words
        composable_admissible_words = [[[chord] for chord in self.chords]]
        # dynamic algo to find words of length n from words of length n-1
        for _ in range(2, len(self.knots)+1):
            largest_length_words = composable_admissible_words[-1]
            new_words = [
                word + [chord] for word in largest_length_words for chord in self.chords
                if word[-1].is_composable_with(chord)]
            # filter on admissibility
            new_words = [word for word in new_words if self.word_is_admissible(word, cyclic=False)]
            composable_admissible_words.append(new_words)
        self.composable_admissible_words = composable_admissible_words

    def _set_rsft_generators(self):
        cyclic_composable_admissible_words = []
        for wl in range(len(self.composable_admissible_words)):
            for w in self.composable_admissible_words[wl]:
                if w[-1].is_composable_with(w[0]):
                    capping_paths = None
                    if self.rsft_graded_by.is_ZZ:
                        if len(w) == 1:
                            capping_paths = [self.get_capping_path(w[0], w[0])]
                        else:
                            capping_paths = [self.get_capping_path(w[i], w[i+1]) for i in range(len(w) - 1)]
                            capping_paths.append(self.get_capping_path(w[-1], w[0]))
                    cyclic_composable_admissible_words.append(
                        RSFTGenerator(
                            word=w,
                            graded_by=self.rsft_graded_by,
                            capping_paths=capping_paths
                        )
                    )
        self.rsft_generators = cyclic_composable_admissible_words

    def _set_rsft_dga(self):
        disks = self.disks
        rsft_del = {w: 0 for w in self.rsft_generators}

        # Helper methods for dealing with extensions of disks to admissible cyclic words

        class DiskExtension(object):
            """Extension of a holomorphic disk obtained by attaching RSFT generators to negative punctures"""

            def __init__(self, disk, output_gens=[]):
                """
                :param disk: Disk instance
                :param gens: list of RSFT generators
                """
                self.disk = disk
                self.output_gens = output_gens
                self._set_disk_asymptotics()
                self._set_asymptotics()
                self._set_first_neg_chord()

            def positive(self):
                return self.n_neg_punctures == 0

            def partially_admissible(self):
                to_knot_labels = []
                for i in range(len(self.asymptotics)):
                    j = (i + 1) % len(self.asymptotics)
                    current_a = self.asymptotics[i]
                    next_a = self.asymptotics[j]
                    if current_a['pos_neg'] == '+':
                        current_to_label = current_a['to_knot_label']
                        to_knot_labels.append(current_to_label)
                        if next_a['pos_neg'] == '+':
                            # False if consecutive positive punctures are not composable
                            if current_to_label != next_a['from_knot_label']:
                                return False
                return utils.is_nonrepeating(to_knot_labels)

            def get_word(self):
                return [a['chord'] for a in self.asymptotics]

            def _set_disk_asymptotics(self):
                self.disk_asymptotics = [
                    {
                        'chord': dc.chord,
                        'pos_neg': dc.pos_neg,
                        'to_knot_label': dc.chord.top_line_segment.knot_label,
                        'from_knot_label': dc.chord.bottom_line_segment.knot_label
                    }
                    for dc in self.disk.disk_corners
                ]
                self.disk_n_neg_punctures = len([a for a in self.disk_asymptotics if a['pos_neg'] == '-'])

            def _set_asymptotics(self):
                asymptotics = []
                temp_output_words = [g.word for g in self.output_gens]
                for a in self.disk_asymptotics:
                    if a['pos_neg'] == '+':
                        asymptotics.append(a)
                    else:
                        if a['chord'] in [w[0] for w in temp_output_words]:
                            new_word = [w for w in temp_output_words if w[0] == a['chord']][0]
                            for chord in new_word[1:]:
                                asymptotics.append({
                                    'chord': chord,
                                    'pos_neg': '+',
                                    'to_knot_label': chord.top_line_segment.knot_label,
                                    'from_knot_label': chord.bottom_line_segment.knot_label
                                })
                            temp_output_words = temp_output_words[1:]
                        else:
                            asymptotics.append(a)
                self.asymptotics = asymptotics
                self.n_neg_punctures = len([a for a in self.asymptotics if a['pos_neg']=='-'])

            def _set_first_neg_chord(self):
                if self.positive():
                    self.first_neg_index = None
                    self.first_neg_chord = None
                    return
                for i in range(len(self.asymptotics)):
                    a = self.asymptotics[i]
                    if a['pos_neg'] == '-':
                        self.first_neg_index = i
                        self.first_neg_chord = a['chord']
                        return

        for d in disks:
            # Create a list of possible extensions to cyclic words of chords
            # while keeping track of the words we use to extend.
            extensions = [DiskExtension(d, output_gens=[])]
            while not all([de.positive() for de in extensions]):
                updated_extensions = []
                for ext in extensions:
                    if ext.partially_admissible():
                        if ext.positive():
                            updated_extensions.append(ext)
                        else:
                            new_generators = [
                                g for g in self.rsft_generators
                                if g.word[0] == ext.first_neg_chord
                            ]
                            for new_gen in new_generators:
                                new_ext = DiskExtension(d, output_gens=ext.output_gens + [new_gen])
                                updated_extensions.append(new_ext)
                extensions = [ext for ext in updated_extensions if ext.partially_admissible()]
            # Now filter for those extensions which are RSFT generators
            extensions = [
                ext for ext in extensions
                if self.get_rsft_generator_from_word(ext.get_word(), raise_if_not_found=False) is not None
            ]
            # At this point we have all extensions of the disk to cyclic words of chords.
            # Now we want to cyclically rotate the input words in all possible ways
            # and add these to our differentials.
            for ext in extensions:
                for i in range(len(ext.asymptotics)):
                    input_word = utils.rotate([a['chord'] for a in ext.asymptotics], i)
                    first_input_chord = input_word[0]
                    input_generator = self.get_rsft_generator_from_word(input_word)
                    if len(ext.output_gens) == 0:
                        # if there are no negative chords, then we contribute 1 to the differential
                        rsft_del[input_generator] += 1
                    else:
                        if first_input_chord in [a['chord'] for a in ext.disk_asymptotics if a['pos_neg'] == '+']:
                            # if the first chord of the input word is in the holo disk,
                            # get the appropriate order
                            index_in_jdisk = [a['chord'] for a in ext.disk_asymptotics].index(first_input_chord)
                            next_negative_puncture_index = index_in_jdisk
                            while ext.disk_asymptotics[next_negative_puncture_index]['pos_neg'] == '+':
                                next_negative_puncture_index += 1
                                next_negative_puncture_index %= len(ext.disk.disk_corners)
                            next_negative_chord = ext.disk_asymptotics[next_negative_puncture_index]['chord']
                            # Here we use the fact that in a plat diagram, each chord can appear at most once
                            # at a negative puncture of an index one disk.
                            output_w_next_neg = [g for g in ext.output_gens if next_negative_chord in g.word][0]
                            index_of_first_output_gen = ext.output_gens.index(output_w_next_neg)
                            output_gens = utils.rotate(ext.output_gens, index_of_first_output_gen)
                        else:
                            # if the first chord in the input word is in one of the output words...
                            first_output_gen = [g for g in ext.output_gens if first_input_chord in g.word][0]
                            index_of_first_output_gen = ext.output_gens.index(first_output_gen)
                            output_words = utils.rotate([g.word for g in ext.output_gens], index_of_first_output_gen)
                            # rotate the first output word so that the first_input_chord leads
                            first_output_word = output_words[0]
                            index_of_chord_in_first_word = first_output_word.index(first_input_chord)
                            first_output_word = utils.rotate(first_output_word, index_of_chord_in_first_word)
                            output_words[0] = first_output_word
                            output_gens = [self.get_rsft_generator_from_word(w) for w in output_words]
                        output_symbols = [g.symbol for g in output_gens]
                        output_monomial = prod(output_symbols)
                        rsft_del[input_generator] += output_monomial
        rsft_del = {
            g.symbol: algebra.Differential(rsft_del[g])
            for g in rsft_del.keys()
        }
        gradings = {g.symbol: g.grading for g in self.rsft_generators}
        self.rsft_dga = algebra.DGA(gradings=gradings, differentials=rsft_del, coeff_mod=2)
