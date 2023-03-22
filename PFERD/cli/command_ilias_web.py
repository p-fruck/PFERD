import argparse
import configparser

from .ilias_common import ilias_common_load, configure_common_group_args
from .parser import CRAWLER_PARSER, SUBPARSERS, load_crawler
from ..logging import log

_PARSER_NAME = "ilias-web"

SUBPARSER = SUBPARSERS.add_parser(
    _PARSER_NAME,
    parents=[CRAWLER_PARSER],
)

GROUP = SUBPARSER.add_argument_group(
    title=f"{_PARSER_NAME} crawler arguments",
    description=f"arguments for the '{_PARSER_NAME}' crawler",
)

GROUP.add_argument(
    "--ilias-url",
    type=str,
    metavar="ILIAS",
    help="The base url of the ilias instance"
)

configure_common_group_args(GROUP)


def load(args: argparse.Namespace, parser: configparser.ConfigParser) -> None:
    log.explain(f"Creating config for command '{_PARSER_NAME}'")

    parser["crawl:ilias"] = {}
    section = parser["crawl:ilias"]
    load_crawler(args, section)

    section["type"] = _PARSER_NAME
    section["ilias_url"] = str(args.ilias_url)

    ilias_common_load(section, args, parser)


SUBPARSER.set_defaults(command=load)
