import argparse
import configparser

from .ilias_common import ilias_common_load, configure_common_group_args
from .parser import CRAWLER_PARSER, SUBPARSERS, load_crawler
from ..logging import log

_PARSER_NAME = "kit-ilias-web"

SUBPARSER = SUBPARSERS.add_parser(
    _PARSER_NAME,
    parents=[CRAWLER_PARSER],
)

GROUP = SUBPARSER.add_argument_group(
    title=f"{_PARSER_NAME} crawler arguments",
    description=f"arguments for the '{_PARSER_NAME}' crawler",
)

configure_common_group_args(GROUP)


def load(args: argparse.Namespace, parser: configparser.ConfigParser) -> None:
    log.explain(f"Creating config for command '{_PARSER_NAME}'")

    parser["crawl:ilias"] = {}
    section = parser["crawl:ilias"]
    load_crawler(args, section)

    section["type"] = _PARSER_NAME

    ilias_common_load(section, args, parser)


SUBPARSER.set_defaults(command=load)
