from argparse import ArgumentParser, ArgumentTypeError


def get_args_parser():
    """Common Args needed for different Learn to Rank training method.
    :rtype: ArgumentParser
    """
    parser = ArgumentParser(description="additional training specification")
    parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)

    return parser
