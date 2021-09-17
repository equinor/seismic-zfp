import click
from seismic_zfp.conversion import SegyConverter, ZgyConverter, SgzConverter

cropping_param_help = (
    "Cropping parameter to apply to input seismic cube. "
    "Refers to IL/XL ordinals rather than numbers."
)
segyconverter_options = [
    click.option(
        "--min-il",
        type=click.INT,
        help=cropping_param_help,
    ),
    click.option(
        "--max-il",
        type=click.INT,
        help=cropping_param_help,
    ),
    click.option(
        "--min-xl",
        type=click.INT,
        help=cropping_param_help,
    ),
    click.option(
        "--max-xl",
        type=click.INT,
        help=cropping_param_help,
    ),
]
sgz_options = [
    click.option(
        "--bits-per-voxel",
        default=4,
        type=click.INT,
        help=(
            "The number of bits to use for storing each seismic voxel. "
            "Recommended using 4-bit, giving 8:1 compression. "
            "Negative value implies reciprocal: i.e. -2 ==> 1/2 bits per voxel"
        ),
        show_default=True,
    ),
    click.option(
        "--blockshape",
        type=click.Tuple([click.INT] * 3),
        default=(4, 4, -1),
        help=(
            "The physical shape of voxels compressed to one disk block. "
            "Can only specify 3 of blockshape (il,xl,z) and bits_per_voxel, "
            "4th is redundant."
        ),
        show_default=True,
    ),
    click.option(
        "--reduce-iops",
        default=False,
        type=click.BOOL,
        help=(
            "Flag to indicate whether compression should attempt to minimize "
            "the number of iops required to read the input SEG-Y file by "
            "reading whole inlines including headers in one go. "
            "Falls back to segyio if incorrect. Useful under Windows."
        ),
        show_default=True,
    ),
]
zgy_options = [
    click.option(
        "--bits-per-voxel",
        default=4,
        type=click.INT,
        help=(
            "The number of bits to use for storing each seismic voxel. "
            "Recommended using 4-bit, giving 8:1 compression. "
            "Negative value implies reciprocal: i.e. -2 ==> 1/2 bits per voxel"
        ),
        show_default=True,
    ),
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@click.group()
def cli():
    """A simple command line interface for seismic-zfp."""


@cli.command("sgy2sgz", short_help="convert a SEGY file to SGZ")
@click.argument(
    "input-segy-file",
    required=True,
    type=click.Path(exists=True),
)
@click.argument(
    "output-sgz-file",
    required=True,
    type=click.Path(),
)
@add_options(sgz_options)
@add_options(segyconverter_options)
def sgy2sgz(
    input_segy_file=None,
    output_sgz_file=None,
    bits_per_voxel=None,
    blockshape=None,
    reduce_iops=None,
    min_il=None,
    max_il=None,
    min_xl=None,
    max_xl=None,
):
    click.echo("Converting {} to {}...".format(input_segy_file, output_sgz_file))
    with SegyConverter(
        input_segy_file,
        min_il=min_il,
        max_il=max_il,
        min_xl=min_xl,
        max_xl=max_xl,
    ) as converter:
        converter.run(
            output_sgz_file,
            bits_per_voxel=bits_per_voxel,
            blockshape=blockshape,
            reduce_iops=reduce_iops,
        )


@cli.command("zgy2sgz", short_help="convert a ZGY file to SGZ")
@click.argument(
    "input-zgy-file",
    required=True,
    type=click.Path(exists=True),
)
@click.argument(
    "output-sgz-file",
    required=True,
    type=click.Path(),
)
@add_options(zgy_options)
def zgy2sgz(
    input_zgy_file=None,
    output_sgz_file=None,
    bits_per_voxel=None,
):
    click.echo("Converting {} to {}...".format(input_zgy_file, output_sgz_file))
    with ZgyConverter(input_zgy_file) as converter:
        converter.run(
            output_sgz_file,
            bits_per_voxel=bits_per_voxel,
        )

@cli.command("sgz2sgy", short_help="convert a SGZ file to SEG-Y")
@click.argument(
    "input-sgz-file",
    required=True,
    type=click.Path(exists=True),
)
@click.argument(
    "output-sgy-file",
    required=True,
    type=click.Path(),
)
def sgz2sgy(
    input_sgz_file=None,
    output_sgy_file=None,
):
    click.echo("Converting {} to {}...".format(input_sgz_file, output_sgy_file))
    with SgzConverter(input_sgz_file) as converter:
        converter.convert_to_segy(output_sgy_file,
        )

if __name__ == "__main__":
    cli()
