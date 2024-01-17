# coding=utf-8
import os
from flask import (
    Flask,
    make_response,
    request,
    redirect,
    url_for)
from jinja2 import Environment, FileSystemLoader, Template
import utils
import legendrian_links as ll


LOG = utils.get_logger(__name__)

# Link resources
LINKS = utils.LINKS
DEFAULT_LINK_KEY = 'm10_132_nv'

# Static resources
STATIC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'static')
JINJA_ENV = Environment(loader=FileSystemLoader(STATIC_PATH))
INDEX_TEMPLATE = JINJA_ENV.get_template('index.html')

# App resources
APP = Flask(__name__)

KNOT_COLORS = [
    (255, 0, 0),
    (128, 0, 128),
    (0, 0, 255),
    (0, 128, 128),
    (0, 255, 0),
    (128, 128, 0)
]
KNOT_ORIENTATIONS_TO_ARROW = {
    'l': '⏪',
    'r': '⏩'
}


def get_template_context(pd, skip_augs=False):
    chords = [
        {
            "string": str(chord),
            "from_knot": chord.bottom_line_segment.knot_label,
            "to_knot": chord.top_line_segment.knot_label
        }
        for chord in pd.chords]
    dgas = []
    if pd.lch_dga is not None:
        dgas.append(get_dga_context(pd.lch_dga, name="LCH", skip_augs=skip_augs))
    if pd.rsft_dga is not None:
        dgas.append(get_dga_context(pd.rsft_dga, name="RSFT", skip_augs=skip_augs))
    template_context = {
        'front_crossings': ','.join([str(c) for c in pd.front_crossings]),
        'svg_context': get_diagram_context(pd),
        'chords': chords,
        'dgas': dgas
    }
    return template_context


def get_diagram_context(pd, increment=50, pad=10):
    knots = pd.knots
    for k in range(len(knots)):
        knots[k]["label"] = k
        knots[k]["rgb"] = KNOT_COLORS[k % len(KNOT_COLORS)]
    n_strands = pd.n_strands
    n_segments = pd.max_x_right
    height = increment * n_strands + pad
    width = (increment * n_segments) + pad + increment
    line_segments = pd.get_line_segment_array()
    output = {
        "knots": knots,
        "knot_labels": list(range(len(knots))),
        "link_is_connected": pd.link_is_connected,
        "linking_matrix": pd.linking_matrix,
        "pad": pad,
        "increment": increment,
        "height": height,
        "width": width,
        "lines": [
            {
                'start_xy': [
                    increment + pad + int(increment * ls['array'][0][0]), pad + int(increment * ls['array'][0][1])],
                'end_xy': [
                    increment + pad + int(increment * ls['array'][1][0]), pad + int(increment * ls['array'][1][1])],
                'rgb': KNOT_COLORS[ls['knot_label'] % len(KNOT_COLORS)],
                'label': {
                    'x': increment + pad + int(increment * (ls['array'][0][0] + ls['array'][1][0]) / 2),
                    'y': pad + int(increment * (ls['array'][0][1] + ls['array'][1][1]) / 2),
                    'marker': KNOT_ORIENTATIONS_TO_ARROW[ls['orientation']] if ls['t_label'] else ''
                }
            }
            for ls in line_segments
        ],
        'x_labels': [{"label": x, "x": int(increment + increment * x + increment / 2)} for x in range(n_segments)],
        'y_labels': [{"label": y, "y": int(pad + increment * y + increment / 2)} for y in range(n_strands - 1)],
    }
    return output


def get_dga_context(dga, name, skip_augs=False):
    output = dict()
    output["name"] = name
    output["grading_mod"] = dga.grading_mod
    output["coeff_mod"] = dga.coeff_mod
    generators = [
        {
            "symbol": g,
            "name": str(g),
            "grading": dga.gradings[g],
            "del": str(dga.differentials[g])
        }
        for g in dga.symbols
    ]
    generators = sorted(generators, key=lambda g: g["name"])
    output["generators"] = generators
    # This variable tells us if we should include information about augmentations/bilinearization
    output["skip_augs"] = skip_augs
    if not skip_augs:
        output["lin_poly_list"] = dga.lin_poly_list
        output["bilin_poly_list"] = dga.bilin_poly_list
        output["n_augs"] = len(dga.augmentations)
        output["has_augs"] = len(dga.augmentations) > 0
        deg_0_gens = [g for g in generators if g["grading"] == 0]
        output["deg_0_gens"] = deg_0_gens
        output["augs"] = [{g["name"]: aug[g["symbol"]] for g in deg_0_gens} for aug in dga.augmentations]
        output["bilin_polys"] = dga.bilin_polys
        output["bilin_polys_dual"] = dga.bilin_polys_dual
    return output


@APP.route('/')
def home():
    n_strands = request.args.get('n_strands')
    if n_strands is not None:
        n_strands = int(n_strands)
        crossings = request.args.get('crossings')
        if crossings is None:
            crossings = []
        else:
            crossings = [int(x) for x in crossings.split(",")]
        # Mirroring
        mirror = False
        mirror_flag = request.args.get('mirror')
        if mirror_flag is not None:
            mirror = mirror_flag.lower() == 'true'
        # Possibly modify orientations of link components
        orientation_flips = None
        orientations = request.args.get('orientation_flips')
        if orientations is not None:
            orientation_flips = [x.lower() == 'true' for x in orientations.split(',')]
        # Take a multiple copy
        n_copy = request.args.get('n_copy')
        if n_copy is not None:
            n_copy = int(n_copy)
        else:
            n_copy = 1
        # Display LCH diff with signs? If true, disregard other flags
        lch_signs = request.args.get('lch_signs')
        if lch_signs is not None:
            lch_signs = lch_signs.lower() == 'true'
        if lch_signs:
            LOG.info("Displaying LCH signs only")
            # This is going to be broken...
            pd = ll.PlatDiagram(
                n_strands=n_strands,
                front_crossings=crossings,
                n_copy=n_copy,
                mirror=mirror,
                orientation_flips=orientation_flips,
                lazy_disks=False,
                lazy_lch=True,
                lazy_rsft=True,
                aug_fill_na=False,
                spec_poly=False
            )
            pd.set_lch(lazy_augs=True, lazy_bilin=True, coeff_mod=0)
            template_context = get_template_context(pd=pd, skip_augs=True)
            LOG.info(pd.lch_dga.differentials)
        else:
            # Manage computation of algebraic invariants
            lazy_lch = True
            lazy_rsft = True
            auto_dga = request.args.get('auto_dgas')
            if auto_dga is not None:
                auto_dgas = auto_dga.lower().split(',')
                if 'lch' in auto_dgas:
                    lazy_lch = False
                if 'rsft' in auto_dgas:
                    lazy_rsft = False
            lazy_disks = False
            lazy_disks_flag = request.args.get('lazy_disks')
            if lazy_disks_flag is not None:
                lazy_disks = lazy_disks_flag.lower() == 'true'
            aug_fill_na = request.args.get('aug_fill_na')
            if aug_fill_na is not None:
                aug_fill_na = int(aug_fill_na)
            spec_poly = False
            spec_poly_flag = request.args.get('spec_poly')
            if spec_poly_flag is not None:
                spec_poly = spec_poly_flag.lower() == 'true'
            pd = ll.PlatDiagram(
                n_strands=n_strands,
                front_crossings=crossings,
                n_copy=n_copy,
                mirror=mirror,
                orientation_flips=orientation_flips,
                lazy_disks=lazy_disks,
                lazy_lch=lazy_lch,
                lazy_rsft=lazy_rsft,
                aug_fill_na=aug_fill_na,
                spec_poly=spec_poly
            )
            template_context = get_template_context(pd=pd)
    else:
        # Display a default link
        data = LINKS[DEFAULT_LINK_KEY].copy()
        data.pop('comment')
        data['lazy_lch'] = True
        data['lazy_rsft'] = False
        pd = ll.PlatDiagram(**data)
        template_context = get_template_context(pd=pd)
    context = INDEX_TEMPLATE.render(**template_context)
    return make_response(context)


if __name__ == "__main__":
    APP.run()
