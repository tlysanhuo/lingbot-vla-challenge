"""Conservative episode-bound sampling of released seen instruction templates.

No simulator, language model, or unseen pool participates. Entity descriptions
are copied verbatim from the original instruction; ambiguous matches retain it.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")


def seen_catalog(task_directory):
    catalog = {}
    for path in sorted(Path(task_directory).glob('*.json')):
        seen = json.loads(path.read_text()).get('seen', [])
        if seen:
            catalog[path.stem] = list(dict.fromkeys(seen))
    return catalog


def catalog_digest(catalog):
    return hashlib.sha256(json.dumps(catalog, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def compile_template(template):
    names = set()
    pieces = []
    pos = 0
    for match in PLACEHOLDER.finditer(template):
        pieces.append(re.escape(template[pos:match.start()]))
        name = match[1]
        pieces.append(f'(?P={name})' if name in names else f'(?P<{name}>.+?)')
        names.add(name)
        pos = match.end()
    pieces.append(re.escape(template[pos:]))
    return re.compile('^' + ''.join(pieces) + '$', re.IGNORECASE)


def arm_references(text):
    return set(re.findall(r'\b(?:left|right|both)\s+arms?\b', text.lower()))


# Slot types transcribed from the released task schemas. None means any
# released object description; literal block/color slots are checked separately.
SLOT_TYPES = {
 'adjust_bottle': {'A':'bottle'}, 'beat_block_hammer': {'A':'hammer'},
 'click_alarmclock': {'A':'alarm-clock'}, 'click_bell': {'A':'bell|callbell'},
 'dump_bin_bigbin': {'A':'tabletrashbin|dustbin'}, 'grab_roller': {'A':'roller'},
 'handover_block': {}, 'handover_mic': {'A':'microphone'},
 'hanging_mug': {'A':'mug','B':'rack'}, 'lift_pot': {'A':'kitchenpot'},
 'move_can_pot': {'A':'kitchenpot','B':'can'}, 'move_pillbottle_pad': {'A':'pillbottle'},
 'move_playingcard_away': {'A':'playingcards'}, 'move_stapler_pad': {'A':'stapler','B':'color'},
 'open_laptop': {'A':'laptop'}, 'open_microwave': {'A':'microwave'},
 'pick_diverse_bottles': {'A':'bottle','B':'bottle'}, 'pick_dual_bottles': {'A':'bottle','B':'bottle'},
 'place_a2b_left': {'A':None,'B':None}, 'place_a2b_right': {'A':None,'B':None},
 'place_bread_basket': {'A':'breadbasket|basket','B':'bread','C':'bread'},
 'place_bread_skillet': {'A':'skillet','B':'bread'},
 'place_burger_fries': {'A':'hamburg','B':'tray','C':'french-fries'},
 'place_can_basket': {'A':'can','B':'basket|breadbasket'},
 'place_cans_plasticbox': {'A':'can','B':'plasticbox','C':'can'},
 'place_container_plate': {'A':'plate','B':None}, 'place_dual_shoes': {'A':'shoe','B':'shoe-box'},
 'place_empty_cup': {'A':'cup','B':'coaster'}, 'place_fan': {'A':'fan','B':'color'},
 'place_mouse_pad': {'A':'mouse','B':'color'},
 'place_object_basket': {'A':None,'B':'basket|breadbasket'},
 'place_object_scale': {'A':'electronicscale','B':None},
 'place_object_stand': {'A':None,'B':'displaystand'},
 'place_phone_stand': {'A':'phone','B':'phonestand'}, 'place_shoe': {'A':'shoe'},
 'press_stapler': {'A':'stapler'},
 'put_bottles_dustbin': {'A':'bottle','B':'bottle','C':'bottle','D':'dustbin'},
 'put_object_cabinet': {'A':None,'B':'cabinet'}, 'rotate_qrcode': {'A':'board|paymentsign'},
 'scan_object': {'A':None,'B':'scanner'}, 'shake_bottle': {'A':'bottle'},
 'shake_bottle_horizontally': {'A':'bottle'},
 'stack_bowls_three': {'A':'bowl'}, 'stack_bowls_two': {'A':'bowl','B':'bowl'},
 'stamp_seal': {'A':'seal','B':'color'}, 'turn_switch': {'A':'switch'},
}
for _task in ('blocks_ranking_rgb', 'stack_blocks_three', 'stack_blocks_two'):
    SLOT_TYPES[_task] = {'A':'=red block','B':'=green block','C':'=blue block'}
SLOT_TYPES['blocks_ranking_size'] = {'A':'=large block','B':'=medium block','C':'=small block'}


def seen_entities(object_directory):
    result = {}
    root = Path(object_directory)
    for path in sorted(root.rglob('*.json')):
        category = path.relative_to(root).parts[0].split('_', 1)[-1]
        for description in json.loads(path.read_text()).get('seen', []):
            for text in (description, 'the ' + description):
                result.setdefault(text.casefold(), set()).add(category)
    colors = set()
    for path in sorted((root.parent.parent / 'envs').glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Dict): continue
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == 3
                    and all(isinstance(x, ast.Constant) and isinstance(x.value, (int, float)) for x in value.elts)):
                    colors.add(key.value.casefold())
    result['__color_names__'] = colors
    return {k: sorted(v) for k, v in result.items()}


def semantic_signature(template):
    # Prevent a generic template match from adding a new spatial/orientation goal.
    literal = PLACEHOLDER.sub('', template).lower()
    signature = set()
    groups = {
        'left': r'\bleft\b', 'right': r'\bright\b',
        'horizontal': r'\bhorizontal(?:ly)?\b', 'vertical': r'\bvertical(?:ly)?\b',
        'upright': r'\bupright\b|head[- ]?up', 'inverted': r'upside[- ]down|head[- ]down',
        'on': r'\b(?:on|onto|upon|atop|above)\b', 'in': r'\b(?:in|into|inside)\b',
        'under': r'\b(?:under|below|beneath)\b', 'next': r'\bnext to\b',
        'clockwise': r'\bclockwise\b', 'counterclockwise': r'\bcounterclockwise\b',
    }
    for name, pattern in groups.items():
        if re.search(pattern, literal): signature.add(name)
    return signature


class SeenBindingBuilder:
    def __init__(self, catalog, entities=None):
        self.catalog = catalog
        self.entities = entities
        self.patterns = [(task, template, compile_template(template))
                         for task, templates in catalog.items() for template in templates]

    def valid_bindings(self, task, values):
        for key, value in values.items():
            if key.islower():
                if value.casefold() not in ('the left arm', 'the right arm', 'the dual arm'):
                    return False
                continue
            if self.entities is None:
                continue
            rules = SLOT_TYPES.get(task, {})
            if key not in rules:
                return False
            rule = rules[key]
            if rule == 'color':
                if value.casefold() not in self.entities.get('__color_names__', []): return False
                continue
            if rule and rule.startswith('='):
                if value.casefold() != rule[1:]: return False
                continue
            types = set(self.entities.get(value.casefold(), []))
            if not types or (rule and not types.intersection(rule.split('|'))):
                return False
        return True

    def bind(self, original):
        matches = {}
        for task, template, pattern in self.patterns:
            match = pattern.fullmatch(original)
            if match:
                bindings = match.groupdict()
                if not self.valid_bindings(task, bindings):
                    continue
                key = (task, tuple(sorted(bindings.items())))
                matches.setdefault(key, (task, template, bindings))
        fallback = {'original': original, 'candidates': [original]}
        if not matches:
            return {**fallback, 'status': 'unmatched'}
        if len(matches) != 1:
            return {**fallback, 'status': 'ambiguous', 'match_count': len(matches)}
        task, source_template, bindings = next(iter(matches.values()))
        candidates = [original]
        # Both bottle tasks use literal arm roles and can share sentence forms;
        # without original scene metadata we cannot prove the A/B-to-arm mapping.
        if task in ('pick_dual_bottles', 'pick_diverse_bottles'):
            return {**fallback, 'status': 'literal_arm_roles_unresolved'}
        if not any(k.islower() for k in bindings) and {'left', 'right'}.issubset(semantic_signature(source_template)):
            return {**fallback, 'status': 'literal_arm_roles_unresolved'}
        entity_order = tuple(dict.fromkeys(k for k in PLACEHOLDER.findall(source_template) if k.isupper()))
        for template in self.catalog[task]:
            # Do not add an unbound arm/object or drop a bound semantic slot.
            if set(PLACEHOLDER.findall(template)) != set(bindings):
                continue
            if tuple(dict.fromkeys(k for k in PLACEHOLDER.findall(template) if k.isupper())) != entity_order:
                continue
            if semantic_signature(template) != semantic_signature(source_template):
                continue
            candidate = PLACEHOLDER.sub(lambda m: bindings[m[1]], template)
            if arm_references(candidate) != arm_references(original):
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        return {'original': original, 'candidates': candidates, 'status': 'bound',
                'task': task, 'source_template': source_template, 'bindings': bindings}


def episode_digest(episodes):
    payload = [(int(e['episode_index']), list(e['tasks']), int(e['length'])) for e in episodes]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


class SeenInstructionBindings:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        if data.get('schema_version') != 1 or data.get('source_pool') != 'release_seen_only':
            raise ValueError('unsupported seen binding manifest')
        self.episodes = data['episodes']
        self.metadata_digest = data['episode_metadata_sha256']
        self.validated = False

    def validate_episodes(self, episodes):
        if episode_digest(episodes) != self.metadata_digest:
            raise ValueError('seen binding manifest does not match the episode metadata')
        self.validated = True

    def sample(self, episode_index, original, generator=None):
        if not self.validated:
            raise RuntimeError('validate episode bindings before use')
        record = self.episodes.get(str(int(episode_index)))
        if record is None or record['original'] != original:
            raise ValueError(f'episode {episode_index}: original instruction mismatch')
        candidates = record['candidates']
        if not candidates or original not in candidates:
            raise ValueError('binding must retain the original instruction')
        if len(candidates) == 1:
            return original
        import torch
        return candidates[int(torch.randint(len(candidates), (), generator=generator))]
