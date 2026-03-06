#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to run build_spice_netlists.py") from exc


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def _project_root_from_here() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_cfg_file(project_root: str) -> str:
    local_path = os.path.join(project_root, "configs", "paths.local.yaml")
    if os.path.isfile(local_path):
        return local_path
    return os.path.join(project_root, "configs", "paths.yaml")


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def cfg_get(cfg: dict, dotted_key: str, default=None):
    cur = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# -----------------------------------------------------------------------------
# Class registry
# -----------------------------------------------------------------------------
CLASS_FAMILY: Dict[str, str] = {
    "text": "skip",
    "junction": "skip",
    "crossover": "skip",
    "terminal": "skip",
    "gnd": "ground",
    "vss": "ground",
    "voltage.dc": "vsource_dc",
    "voltage.ac": "vsource_ac",
    "voltage.battery": "vsource_battery",
    "current.dc": "isource_dc",
    "current.ac": "isource_ac",
    "resistor": "resistor",
    "resistor.adjustable": "resistor",
    "resistor.photo": "resistor",
    "capacitor.unpolarized": "capacitor",
    "capacitor.polarized": "capacitor",
    "capacitor.adjustable": "capacitor",
    "inductor": "inductor",
    "inductor.ferrite": "inductor",
    "inductor.coupled": "placeholder",
    "transformer": "placeholder",
    "diode": "diode",
    "diode.light_emitting": "diode_led",
    "diode.thyrector": "placeholder",
    "diode.zener": "diode_zener",
    "diac": "placeholder",
    "triac": "placeholder",
    "thyristor": "placeholder",
    "varistor": "placeholder",
    "transistor.bjt": "bjt",
    "transistor.fet": "mosfet",
    "transistor.photo": "placeholder",
    "operational_amplifier": "opamp",
    "operational_amplifier.schmitt_trigger": "opamp",
    "optocoupler": "placeholder",
    "integrated_circuit": "placeholder",
    "integrated_circuit.ne555": "placeholder",
    "integrated_circuit.voltage_regulator": "placeholder",
    "xor": "placeholder",
    "and": "placeholder",
    "or": "placeholder",
    "not": "placeholder",
    "nand": "placeholder",
    "nor": "placeholder",
    "probe": "skip",
    "probe.current": "skip",
    "probe.voltage": "skip",
    "switch": "placeholder",
    "relay": "placeholder",
    "socket": "placeholder",
    "fuse": "placeholder",
    "speaker": "placeholder",
    "motor": "placeholder",
    "lamp": "placeholder",
    "microphone": "placeholder",
    "antenna": "placeholder",
    "crystal": "placeholder",
    "mechanical": "placeholder",
    "magnetic": "placeholder",
    "optical": "placeholder",
}

CLASS_ALIASES: Dict[str, str] = {
    "current.dc": "current.dc",
    "current.ac": "current.ac",
}

EXPECTED_YOLO_CLASSES: Tuple[str, ...] = (
    "text", "junction", "crossover", "terminal", "gnd", "vss",
    "voltage.dc", "voltage.ac", "voltage.battery", "current.ac", "resistor",
    "resistor.adjustable", "resistor.photo", "capacitor.unpolarized",
    "capacitor.polarized", "capacitor.adjustable", "inductor",
    "inductor.ferrite", "inductor.coupled", "transformer", "diode",
    "diode.light_emitting", "diode.thyrector", "diode.zener", "diac",
    "triac", "thyristor", "varistor", "transistor.bjt",
    "transistor.fet", "transistor.photo", "operational_amplifier",
    "operational_amplifier.schmitt_trigger", "optocoupler",
    "integrated_circuit", "integrated_circuit.ne555",
    "integrated_circuit.voltage_regulator", "xor", "and", "or", "not",
    "nand", "nor", "probe", "probe.current", "probe.voltage", "switch",
    "relay", "socket", "fuse", "speaker", "motor", "lamp",
    "microphone", "antenna", "crystal", "mechanical", "magnetic",
    "optical",
)

missing = [name for name in EXPECTED_YOLO_CLASSES if name not in CLASS_FAMILY]
if missing:  # pragma: no cover
    raise RuntimeError(f"CLASS_FAMILY missing entries: {missing}")


# -----------------------------------------------------------------------------
# Value parsing
# -----------------------------------------------------------------------------
PREFIX_SCALE: Dict[str, float] = {
    "": 1.0,
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

ENG_FMT_ORDER = [
    (1e12, "T"),
    (1e9, "G"),
    (1e6, "MEG"),
    (1e3, "K"),
    (1.0, ""),
    (1e-3, "m"),
    (1e-6, "u"),
    (1e-9, "n"),
    (1e-12, "p"),
    (1e-15, "f"),
]

UNIT_ALIASES: Dict[str, str] = {
    "ω": "ohm",
    "ohms": "ohm",
    "ohm": "ohm",
    "r": "ohm",
    "v": "v",
    "vac": "v",
    "vdc": "v",
    "mv": "v",
    "a": "a",
    "ma": "a",
    "ua": "a",
    "f": "f",
    "uf": "f",
    "nf": "f",
    "pf": "f",
    "h": "h",
    "mh": "h",
    "uh": "h",
    "hz": "hz",
    "khz": "hz",
    "mhz": "hz",
    "ghz": "hz",
}

VALUE_RE = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<num>(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(?P<prefix>meg|[tgkmunpf]?)\s*(?P<unit>[A-Za-z]+)?",
    re.IGNORECASE,
)

RKM_RE = re.compile(
    r"(?P<int>\d+)(?P<rkm>[rRkKmMuUnNpPfF])(?P<frac>\d+)"
)


@dataclass
class ParsedValue:
    raw_text: str
    numeric: float
    unit: str

    @property
    def spice(self) -> str:
        return format_eng(self.numeric)


def normalize_text_value(text: str) -> str:
    s = (text or "").strip()
    s = s.replace("μ", "u").replace("µ", "u")
    s = s.replace("Ω", "ohm").replace("ω", "ohm")
    s = s.replace("−", "-").replace("—", "-")
    s = s.replace("，", ",").replace("；", ";")
    s = s.replace("（", "(").replace("）", ")")
    # OCR confusions in numeric contexts
    s = re.sub(r"(?<=\d)[oO](?=\d)", "0", s)
    s = re.sub(r"(?<=\d)[lI](?=\d)", "1", s)
    s = s.replace("O", "0") if re.search(r"\d[OolI]|[OolI]\d", s) else s
    s = s.replace(",", ".") if re.fullmatch(r"\s*\d+,\d+\s*[A-Za-zΩω]*\s*", s) else s
    s = s.replace(" ", "")
    return s


def format_eng(value: float) -> str:
    if not math.isfinite(value) or value == 0:
        return "0"
    abs_val = abs(value)
    for scale, suffix in ENG_FMT_ORDER:
        if abs_val >= scale or scale == 1e-15:
            x = value / scale
            if abs(x - round(x)) < 1e-12:
                return f"{int(round(x))}{suffix}"
            return f"{x:.6g}{suffix}"
    return f"{value:.6g}"


def _split_text_candidates(raw_texts: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in raw_texts:
        if raw is None:
            continue
        parts = re.split(r"[\s,;/|]+", str(raw).strip())
        out.extend([p for p in parts if p])
    return out


def _parse_rkm_token(token: str) -> Optional[ParsedValue]:
    s = normalize_text_value(token)
    m = RKM_RE.fullmatch(s)
    if not m:
        return None
    int_part = m.group("int")
    frac = m.group("frac")
    mark = m.group("rkm").lower()
    numeric = float(f"{int_part}.{frac}")
    if mark == "r":
        return ParsedValue(raw_text=token, numeric=numeric, unit="ohm")
    if mark in PREFIX_SCALE:
        unit = "ohm" if mark == "k" else ""
        return ParsedValue(raw_text=token, numeric=numeric * PREFIX_SCALE[mark], unit=unit)
    return None


def _expand_compound_unit(token: str) -> str:
    s = normalize_text_value(token)
    s_low = s.lower()
    for prefix, unit in [("meg", ""), ("g", ""), ("k", ""), ("m", ""), ("u", ""), ("n", ""), ("p", ""), ("f", "")]:
        pass
    # Keep common compact forms interpretable by VALUE_RE
    replacements = [
        (r"(?i)(\d)(ma)$", r"\1mA"),
        (r"(?i)(\d)(ua)$", r"\1uA"),
        (r"(?i)(\d)(mv)$", r"\1mV"),
        (r"(?i)(\d)(khz)$", r"\1kHz"),
        (r"(?i)(\d)(mhz)$", r"\1MEGHzTMP"),
    ]
    for pat, rep in replacements:
        s = re.sub(pat, rep, s)
    s = s.replace("MEGHzTMP", "MHz")
    return s


def _unit_with_prefix_scale(unit_raw: str) -> Tuple[str, float]:
    u = (unit_raw or "").lower()
    if not u:
        return "", 1.0
    if u in {"v", "a", "f", "h", "hz", "ohm"}:
        return u, 1.0
    # combined unit forms: mA, uF, kHz, mV, etc.
    for prefix in ["meg", "g", "k", "m", "u", "n", "p", "f"]:
        if u.startswith(prefix) and u[len(prefix):] in {"v", "a", "f", "h", "hz", "ohm"}:
            return u[len(prefix):], PREFIX_SCALE[prefix]
    mapped = UNIT_ALIASES.get(u)
    if mapped and mapped != u:
        return _unit_with_prefix_scale(u)
    return UNIT_ALIASES.get(u, u), 1.0


def parse_value_candidates(raw_texts: Sequence[str]) -> List[ParsedValue]:
    out: List[ParsedValue] = []
    seen = set()

    for raw in raw_texts:
        if raw is None:
            continue
        tokens = [str(raw)] + _split_text_candidates([str(raw)])
        for token in tokens:
            token = _expand_compound_unit(token)
            if not token:
                continue

            rkm = _parse_rkm_token(token)
            if rkm is not None:
                key = (round(rkm.numeric, 18), rkm.unit)
                if key not in seen:
                    seen.add(key)
                    out.append(rkm)
                continue

            s = normalize_text_value(token)
            m = VALUE_RE.search(s)
            if not m:
                continue

            sign = -1.0 if m.group("sign") == "-" else 1.0
            num = float(m.group("num"))
            prefix = (m.group("prefix") or "").lower()
            unit_raw = (m.group("unit") or "")
            unit, unit_scale = _unit_with_prefix_scale(unit_raw)

            # Distinguish bare trailing 'f' as femto prefix vs farad unit.
            if not unit and prefix == "f" and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)f", s, flags=re.IGNORECASE):
                prefix = ""
                unit = "f"

            if prefix not in PREFIX_SCALE:
                continue

            numeric = sign * num * PREFIX_SCALE[prefix] * unit_scale
            pv = ParsedValue(raw_text=raw, numeric=numeric, unit=unit)
            key = (round(pv.numeric, 18), pv.unit)
            if key not in seen:
                seen.add(key)
                out.append(pv)
    return out


def choose_component_value(class_family: str, raw_texts: Sequence[str]) -> Optional[ParsedValue]:
    candidates = parse_value_candidates(raw_texts)
    if not candidates:
        return None

    desired_units = {
        "resistor": {"", "ohm"},
        "capacitor": {"", "f"},
        "inductor": {"", "h"},
        "vsource_dc": {"", "v"},
        "vsource_ac": {"", "v", "hz"},
        "vsource_battery": {"", "v"},
        "isource_dc": {"", "a"},
        "isource_ac": {"", "a", "hz"},
        "diode_zener": {"", "v"},
    }.get(class_family, set())

    if desired_units:
        for cand in candidates:
            if cand.unit in desired_units:
                return cand
    return candidates[0]


@dataclass
class ACSourceParams:
    amplitude: Optional[ParsedValue]
    frequency: Optional[ParsedValue]
    source_kind: str  # "voltage", "current", or "unknown"


def parse_ac_source_params(raw_texts: Sequence[str]) -> ACSourceParams:
    amp_v_candidates: List[ParsedValue] = []
    amp_i_candidates: List[ParsedValue] = []
    freq_candidates: List[ParsedValue] = []
    unitless_candidates: List[ParsedValue] = []
    all_candidates = parse_value_candidates(raw_texts)

    for cand in all_candidates:
        if cand.unit == "hz":
            freq_candidates.append(cand)
        elif cand.unit == "v":
            amp_v_candidates.append(cand)
        elif cand.unit == "a":
            amp_i_candidates.append(cand)
        elif cand.unit == "":
            unitless_candidates.append(cand)

    # Fallback for OCR strings like "5 3kHz" or "1 2k".
    if not amp_v_candidates and not amp_i_candidates and unitless_candidates:
        amp_guess = unitless_candidates[0]
        if freq_candidates:
            amp_v_candidates.append(amp_guess)
        elif len(unitless_candidates) >= 2:
            amp_v_candidates.append(unitless_candidates[0])
            second = unitless_candidates[1]
            if abs(second.numeric) >= 10:
                freq_candidates.append(ParsedValue(raw_text=second.raw_text, numeric=second.numeric, unit="hz"))

    source_kind = "unknown"
    amplitude = None
    if amp_i_candidates:
        amplitude = amp_i_candidates[0]
        source_kind = "current"
    elif amp_v_candidates:
        amplitude = amp_v_candidates[0]
        source_kind = "voltage"

    frequency = freq_candidates[0] if freq_candidates else None
    return ACSourceParams(amplitude=amplitude, frequency=frequency, source_kind=source_kind)


def infer_source_family(comp: dict, family: str, raw_texts: Sequence[str]) -> str:
    if family not in {"vsource_dc", "vsource_ac", "vsource_battery", "isource_dc", "isource_ac"}:
        return family

    text_blob = " ".join(str(t) for t in raw_texts if t).lower()
    candidates = parse_value_candidates(raw_texts)
    has_a = any(c.unit == "a" for c in candidates) or bool(re.search(r"(?<![a-z])(ma|ua|a)(?![a-z])", text_blob))
    has_v = any(c.unit == "v" for c in candidates) or bool(re.search(r"(?<![a-z])(mv|v)(?![a-z])", text_blob))

    if family == "vsource_ac":
        params = parse_ac_source_params(raw_texts)
        if params.source_kind == "current" or (has_a and not has_v):
            return "isource_ac"
        return family

    if family in {"vsource_dc", "vsource_battery"} and has_a and not has_v:
        return "isource_dc"

    if family in {"isource_dc", "isource_ac"} and has_v and not has_a:
        return "vsource_ac" if family == "isource_ac" else "vsource_dc"

    return family


# -----------------------------------------------------------------------------
# Net / pin utilities
# -----------------------------------------------------------------------------

def canonical_class_name(comp: dict) -> str:
    refined = comp.get("cls_name_refined")
    if isinstance(refined, str) and refined.strip():
        return CLASS_ALIASES.get(refined.strip(), refined.strip())
    raw = str(comp.get("cls_name", "")).strip()
    return CLASS_ALIASES.get(raw, raw)


def family_for_component(comp: dict) -> str:
    cls_name = canonical_class_name(comp)
    return CLASS_FAMILY.get(cls_name, "placeholder")


def family_base_name(comp: dict) -> str:
    cls_name = canonical_class_name(comp)
    return re.sub(r"[^A-Za-z0-9]+", "_", cls_name).strip("_").upper() or "BLOCK"


def is_ground_component(comp: dict) -> bool:
    return family_for_component(comp) == "ground"


def endpoint_sort_key(ep: dict, bbox: Optional[Sequence[float]] = None) -> Tuple[int, float, float, int]:
    side_rank = {"left": 0, "top": 1, "right": 2, "bottom": 3}
    side = str(ep.get("side", "")).lower()
    x = float(ep.get("x", 0.0))
    y = float(ep.get("y", 0.0))
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = map(float, bbox)
        horizontal = (x2 - x1) >= (y2 - y1)
        if horizontal:
            return (side_rank.get(side, 99), x, y, int(ep.get("eid", 0)))
        return (side_rank.get(side, 99), y, x, int(ep.get("eid", 0)))
    return (side_rank.get(side, 99), x, y, int(ep.get("eid", 0)))


def normalize_role(role: Optional[str]) -> str:
    if not role:
        return ""
    s = role.lower().strip()
    s = s.replace("+", "plus").replace("-", "minus")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


ROLE_ORDER_MAP: Dict[str, int] = {
    "positive": 0,
    "plus": 0,
    "negative": 1,
    "minus": 1,
    "flowingfrom": 0,
    "flowingto": 1,
    "anode": 0,
    "cathode": 1,
    "collector": 0,
    "base": 1,
    "emitter": 2,
    "drain": 0,
    "gate": 1,
    "source": 2,
    "bulk": 3,
    "body": 3,
    "substrate": 3,
    "innoninverting": 0,
    "noninverting": 0,
    "inplus": 0,
    "inverting": 1,
    "inminus": 1,
    "output": 2,
    "out": 2,
    "vplus": 3,
    "vcc": 3,
    "vminus": 4,
    "vee": 4,
    "gnd": 5,
    "primary1": 0,
    "primary2": 1,
    "secondary1": 2,
    "secondary2": 3,
}


def order_component_endpoints(comp: dict, endpoints: Sequence[dict]) -> List[dict]:
    bbox = comp.get("bbox")

    def key(ep: dict) -> Tuple[int, int, float, float, int]:
        role_norm = normalize_role(ep.get("port_role"))
        role_rank = ROLE_ORDER_MAP.get(role_norm, 99)
        base = endpoint_sort_key(ep, bbox)
        return (0 if role_rank != 99 else 1, role_rank, base[1], base[2], base[3])

    return sorted(endpoints, key=key)


# -----------------------------------------------------------------------------
# Subcircuits
# -----------------------------------------------------------------------------
@dataclass
class SubcktDef:
    name: str
    kind: str
    pin_count: int

    def render(self) -> List[str]:
        pins = [f"P{i+1}" for i in range(self.pin_count)]

        if self.kind == "placeholder":
            lines = [f".SUBCKT {self.name} {' '.join(pins)}"]
            if self.pin_count <= 1:
                lines.append("* single-pin placeholder")
            else:
                anchor = pins[0]
                for idx, pin in enumerate(pins[1:], start=2):
                    lines.append(f"RSTUB{idx} {anchor} {pin} 1G")
            lines.append(f".ENDS {self.name}")
            return lines

        if self.kind == "opamp3":
            # Pins: IN+, IN-, OUT
            return [
                f".SUBCKT {self.name} P1 P2 P3",
                "RINP P1 0 1G",
                "RINN P2 0 1G",
                "EGAIN NINT 0 P1 P2 1e5",
                "ROUT NINT P3 100",
                "RREF P3 0 100MEG",
                "CPOLE NINT 0 1p",
                f".ENDS {self.name}",
            ]

        if self.kind == "opamp5":
            # Pins: IN+, IN-, OUT, V+, V-
            return [
                f".SUBCKT {self.name} P1 P2 P3 P4 P5",
                "RINP P1 0 1G",
                "RINN P2 0 1G",
                "EGAIN NINT P5 P1 P2 1e5",
                "DCLPH NINT P4 DCLAMP_AUTO",
                "DCLPL P5 NINT DCLAMP_AUTO",
                "ROUT NINT P3 100",
                "RREF P3 P5 100MEG",
                "CPOLE NINT P5 1p",
                f".ENDS {self.name}",
            ]

        return [f".SUBCKT {self.name} {' '.join(pins)}", f".ENDS {self.name}"]


# -----------------------------------------------------------------------------
# Core exporter
# -----------------------------------------------------------------------------

def build_net_lookup(final_js: dict) -> Tuple[Dict[int, str], Dict[int, int]]:
    endpoint_to_netname: Dict[int, str] = {}
    endpoint_to_netid: Dict[int, int] = {}
    component_by_id = {int(c["id"]): c for c in final_js.get("components", []) if "id" in c}

    for net in final_js.get("nets", []):
        net_id = int(net.get("id", -1))
        endpoint_eids = [int(eid) for eid in net.get("endpoint_eids", [])]
        comp_ids = [int(cid) for cid in net.get("component_ids", [])]
        is_gnd = any(is_ground_component(component_by_id[cid]) for cid in comp_ids if cid in component_by_id)
        net_name = "0" if is_gnd else f"N{net_id}"
        for eid in endpoint_eids:
            endpoint_to_netname[eid] = net_name
            endpoint_to_netid[eid] = net_id
    return endpoint_to_netname, endpoint_to_netid


def component_text_map(final_js: dict) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = defaultdict(list)
    for item in final_js.get("texts", []):
        try:
            cid = item.get("matched_component_id")
            if cid is None:
                continue
            out[int(cid)].append(str(item.get("text", "")).strip())
        except Exception:
            continue
    return out


def component_endpoint_map(final_js: dict) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = defaultdict(list)
    for ep in final_js.get("endpoints", []):
        if ep.get("kind") != "terminal":
            continue
        cid = ep.get("comp_id")
        if cid is None:
            continue
        out[int(cid)].append(ep)
    return out


def default_value_for_family(family: str, class_name: str) -> str:
    defaults = {
        "resistor": "1K",
        "capacitor": "1U",
        "inductor": "1M",
        "vsource_dc": "5",
        "vsource_ac": "1",
        "vsource_battery": "9",
        "isource_dc": "1m",
        "isource_ac": "1m",
    }
    return defaults.get(family, "1")


def model_for_bjt(comp: dict) -> str:
    subtype = str(comp.get("subtype", "")).lower()
    refined = str(comp.get("cls_name_refined", "")).lower()
    raw = str(comp.get("cls_name", "")).lower()
    text = " ".join(filter(None, [subtype, refined, raw]))
    return "QPNP_AUTO" if "pnp" in text else "QNPN_AUTO"


def model_for_mosfet(comp: dict) -> str:
    subtype = str(comp.get("subtype", "")).strip().lower()
    refined = str(comp.get("cls_name_refined", "")).strip().lower()
    raw = str(comp.get("cls_name", "")).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", " ".join(filter(None, [subtype, refined, raw]))).strip()

    p_tokens = {"p", "pmos", "pfet", "pchannel", "pch", "pmosfet"}
    n_tokens = {"n", "nmos", "nfet", "nchannel", "nch", "nmosfet"}
    pieces = set(text.split())
    if pieces & p_tokens or any(tag in text for tag in ["p mos", "p channel"]):
        return "PMOS_AUTO"
    if pieces & n_tokens or any(tag in text for tag in ["n mos", "n channel"]):
        return "NMOS_AUTO"
    return "NMOS_AUTO"


def register_subckt(subckt_defs: Dict[str, SubcktDef], name: str, kind: str, pin_count: int) -> None:
    if name not in subckt_defs:
        subckt_defs[name] = SubcktDef(name=name, kind=kind, pin_count=pin_count)


def render_component(
    comp: dict,
    ordered_eps: Sequence[dict],
    endpoint_to_net: Dict[int, str],
    text_values: Sequence[str],
    subckt_defs: Dict[str, SubcktDef],
) -> List[str]:
    comp_id = int(comp.get("id", -1))
    cls_name = canonical_class_name(comp)
    base_family = family_for_component(comp)
    family = infer_source_family(comp, base_family, text_values)
    nets = [endpoint_to_net.get(int(ep["eid"]), f"NC_{ep['eid']}") for ep in ordered_eps]
    value = choose_component_value(family, text_values)
    raw_value_text = ", ".join(text_values) if text_values else ""

    prefix_base = {
        "resistor": "R",
        "capacitor": "C",
        "inductor": "L",
        "vsource_dc": "V",
        "vsource_ac": "V",
        "vsource_battery": "V",
        "isource_dc": "I",
        "isource_ac": "I",
        "diode": "D",
        "diode_led": "D",
        "diode_zener": "D",
        "bjt": "Q",
        "mosfet": "M",
        "opamp": "X",
        "placeholder": "X",
    }.get(family, "X")
    refdes = f"{prefix_base}{comp_id + 1}"

    comment = f"* comp_id={comp_id} class={cls_name}"
    if family != base_family:
        comment += f" inferred_family={family}"
    if raw_value_text:
        comment += f" OCR=[{raw_value_text}]"

    if family == "skip":
        return [f"* skipped symbolic class {cls_name} (comp_id={comp_id})"]
    if family == "ground":
        return [f"* ground symbol {cls_name} (comp_id={comp_id})"]

    lines = [comment]

    def need_pins(min_pins: int, exact: Optional[int] = None) -> bool:
        if exact is not None:
            return len(nets) == exact
        return len(nets) >= min_pins

    if family == "resistor" and need_pins(2, exact=2):
        lines.append(f"{refdes} {nets[0]} {nets[1]} {(value.spice if value else default_value_for_family(family, cls_name))}")
        return lines

    if family == "capacitor" and need_pins(2, exact=2):
        lines.append(f"{refdes} {nets[0]} {nets[1]} {(value.spice if value else default_value_for_family(family, cls_name))}")
        return lines

    if family == "inductor" and need_pins(2, exact=2):
        lines.append(f"{refdes} {nets[0]} {nets[1]} {(value.spice if value else default_value_for_family(family, cls_name))}")
        return lines

    if family == "vsource_dc" and need_pins(2, exact=2):
        dc_val = value.spice if value else default_value_for_family(family, cls_name)
        lines.append(f"{refdes} {nets[0]} {nets[1]} DC {dc_val}")
        return lines

    if family == "vsource_battery" and need_pins(2, exact=2):
        dc_val = value.spice if value else default_value_for_family(family, cls_name)
        lines.append(f"{refdes} {nets[0]} {nets[1]} DC {dc_val}")
        return lines

    if family == "vsource_ac" and need_pins(2, exact=2):
        ac = parse_ac_source_params(text_values)
        amp = ac.amplitude.spice if ac.amplitude else default_value_for_family(family, cls_name)
        freq = ac.frequency.spice if ac.frequency else "1K"
        lines.append(f"{refdes} {nets[0]} {nets[1]} AC {amp} SIN(0 {amp} {freq})")
        return lines

    if family == "isource_dc" and need_pins(2, exact=2):
        dc_val = value.spice if value else default_value_for_family(family, cls_name)
        lines.append(f"{refdes} {nets[0]} {nets[1]} DC {dc_val}")
        return lines

    if family == "isource_ac" and need_pins(2, exact=2):
        ac = parse_ac_source_params(text_values)
        amp = ac.amplitude.spice if ac.amplitude else default_value_for_family(family, cls_name)
        freq = ac.frequency.spice if ac.frequency else "1K"
        lines.append(f"{refdes} {nets[0]} {nets[1]} AC {amp} SIN(0 {amp} {freq})")
        return lines

    if family in {"diode", "diode_led", "diode_zener"} and need_pins(2, exact=2):
        model = {"diode": "DDEFAULT_AUTO", "diode_led": "DLED_AUTO", "diode_zener": "DZENER_AUTO"}[family]
        lines.append(f"{refdes} {nets[0]} {nets[1]} {model}")
        return lines

    if family == "bjt" and need_pins(3):
        nets3 = (nets + ["0", "0", "0"])[:4]
        model = model_for_bjt(comp)
        if len(nets) >= 4:
            lines.append(f"{refdes} {nets3[0]} {nets3[1]} {nets3[2]} {nets3[3]} {model}")
        else:
            lines.append(f"{refdes} {nets3[0]} {nets3[1]} {nets3[2]} {model}")
        return lines

    if family == "mosfet" and need_pins(3):
        nets4 = (nets + [nets[2] if len(nets) >= 3 else "0", "0"])[:4]
        model = model_for_mosfet(comp)
        lines.append(f"{refdes} {nets4[0]} {nets4[1]} {nets4[2]} {nets4[3]} {model}")
        return lines

    if family == "opamp" and need_pins(3):
        pin_count = len(nets)
        if pin_count >= 5:
            subckt_name = "OPAMP_AUTO_5P"
            register_subckt(subckt_defs, subckt_name, "opamp5", 5)
            lines.append(f"X{comp_id + 1} {' '.join(nets[:5])} {subckt_name}")
        else:
            subckt_name = "OPAMP_AUTO_3P"
            register_subckt(subckt_defs, subckt_name, "opamp3", 3)
            lines.append(f"X{comp_id + 1} {' '.join(nets[:3])} {subckt_name}")
        return lines

    pin_count = max(1, len(nets))
    subckt_name = f"{family_base_name(comp)}_AUTO_{pin_count}P"
    register_subckt(subckt_defs, subckt_name, "placeholder", pin_count)
    if nets:
        lines.append(f"X{comp_id + 1} {' '.join(nets)} {subckt_name}")
    else:
        lines.append(f"* placeholder class {cls_name} has no resolved terminals")
    return lines


def render_models() -> List[str]:
    return [
        ".MODEL DDEFAULT_AUTO D",
        ".MODEL DLED_AUTO D(IS=1e-18 N=2)",
        ".MODEL DZENER_AUTO D(BV=5.1 IBV=1m)",
        ".MODEL DCLAMP_AUTO D(IS=1e-15 N=1)",
        ".MODEL QNPN_AUTO NPN",
        ".MODEL QPNP_AUTO PNP",
        ".MODEL NMOS_AUTO NMOS (LEVEL=1 VTO=1 KP=1m)",
        ".MODEL PMOS_AUTO PMOS (LEVEL=1 VTO=-1 KP=1m)",
    ]


def export_one(final_json_path: str, output_dir: str) -> str:
    with open(final_json_path, "r", encoding="utf-8") as f:
        final_js = json.load(f)

    image_name = str(final_js.get("image") or os.path.splitext(os.path.basename(final_json_path))[0].replace(".final", ""))
    endpoint_to_net, _ = build_net_lookup(final_js)
    comp_to_texts = component_text_map(final_js)
    comp_to_eps = component_endpoint_map(final_js)
    subckt_defs: Dict[str, SubcktDef] = {}

    lines: List[str] = [
        f"* Auto-generated from {os.path.basename(final_json_path)}",
        f"* image={image_name}",
        "* Native primitives are emitted directly.",
        "* Higher-level / ambiguous symbols use autogenerated subcircuits.",
        ".TITLE " + image_name,
        ".OPTIONS ABSTOL=1n RELTOL=1e-3",
        "",
        "* ---- Element section ----",
    ]

    components = sorted(final_js.get("components", []), key=lambda c: int(c.get("id", 0)))
    for comp in components:
        cid = int(comp.get("id", -1))
        ordered_eps = order_component_endpoints(comp, comp_to_eps.get(cid, []))
        lines.extend(
            render_component(
                comp=comp,
                ordered_eps=ordered_eps,
                endpoint_to_net=endpoint_to_net,
                text_values=comp_to_texts.get(cid, []),
                subckt_defs=subckt_defs,
            )
        )

    lines.extend(["", "* ---- Device models ----"])
    lines.extend(render_models())

    if subckt_defs:
        lines.extend(["", "* ---- Autogenerated subcircuits ----"])
        for name in sorted(subckt_defs):
            lines.extend(subckt_defs[name].render())
            lines.append("")

    lines.extend(["* ---- End of netlist ----", ".END", ""])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{image_name}.spice")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SPICE] Saved: {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    project_root = _project_root_from_here()
    cfg_path = _default_cfg_file(project_root)
    parser = argparse.ArgumentParser(description="Generate SPICE netlists from final JSON files.")
    parser.add_argument("--cfg", default=cfg_path, help="Path to paths.yaml / paths.local.yaml")
    parser.add_argument("--input-dir", default=None, help="Directory containing *.final.json")
    parser.add_argument("--output-dir", default=None, help="Directory to store generated .spice files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = {}
    if args.cfg and os.path.isfile(args.cfg):
        cfg = load_yaml(args.cfg)
    elif not (args.input_dir and args.output_dir):
        raise FileNotFoundError(f"Config file not found: {args.cfg}")

    final_result_root = cfg_get(cfg, "paths.final_result_root")
    if args.input_dir:
        input_dir = args.input_dir
    else:
        if not final_result_root:
            raise ValueError("Missing paths.final_result_root in config")
        input_dir = os.path.join(str(final_result_root), "json")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = cfg_get(cfg, "paths.spice_netlists")
        if not output_dir:
            if not final_result_root:
                raise ValueError("Missing paths.final_result_root in config")
            output_dir = os.path.join(os.path.dirname(str(final_result_root)), "spice_netlists")

    json_files = sorted(glob(os.path.join(str(input_dir), "*.final.json")))
    if not json_files:
        print(f"[WARN] No final JSON found in: {input_dir}")
        return 0

    print(f"[INFO] input_dir  = {input_dir}")
    print(f"[INFO] output_dir = {output_dir}")
    for path in json_files:
        export_one(path, str(output_dir))
    print(f"[DONE] Generated {len(json_files)} SPICE netlist(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
