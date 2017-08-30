import json
import logging

METADATA_JSON_PATH = 'metadata.json'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_metadata():
    global METADATA_JSON_PATH

    with open(METADATA_JSON_PATH, 'r') as metadata_file:
        metadata_json = json.load(metadata_file)

    return metadata_json


def get_last_patch():
    metadata_json = _load_metadata()
    return metadata_json['patches'][0]


def get_patch(patch_name):
    global logger
    metadata_json = _load_metadata()

    for entry in metadata_json['patches']:
        if entry['patch_name'] == patch_name:
            return entry

    logger.error('Could not find patch %s', patch_name)


def get_hero_dict():
    metadata_json = _load_metadata()
    hero_dict = dict()

    for entry in metadata_json['heroes']:
        hero_dict[entry['id']] = entry['name']

    return hero_dict
