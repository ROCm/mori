#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SlotUsage:
    slot_name: str
    source_file: Path
    line_number: int


@dataclass
class SlotGroup:
    base_name: str
    module_path: str
    profiler_namespace: str
    enum_name: str
    slots: List[str]
    source_files: List[Path]


def camel_to_snake(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def extract_profiler_namespace_hint(cpp_file: Path) -> Optional[str]:
    try:
        content = cpp_file.read_text(encoding="utf-8")
    except (IOError, OSError, UnicodeDecodeError):
        return None

    pattern = r"MORI_DECLARE_PROFILER_CONTEXT\s*\([^,]+,\s*([\w:]+),"
    match = re.search(pattern, content)
    if match:
        full_type = match.group(1)
        parts = full_type.split("::")
        if len(parts) > 1:
            return "::".join(parts[:-1])

    return None


def infer_profiler_namespace(cpp_file: Path, project_root: Path) -> Tuple[str, str]:
    try:
        relative = cpp_file.relative_to(project_root / "src")
    except ValueError:
        relative = cpp_file.parent.name / cpp_file.name

    parts = relative.parts[:-1]

    if not parts:
        module = "core"
    elif parts[0] == "ops":
        module = parts[1] if len(parts) > 1 else "ops"
    # Support subdirectory-based module subdivision (reserved for future use)
    # Example: src/collectives/ibgda/foo.cpp -> module = "ibgda"
    elif parts[0] in ["collectives", "shmem", "io"]:
        module = parts[1] if len(parts) > 1 else parts[0]
    else:
        module = parts[0]

    profiler_ns = f"mori::profiler::{module}"
    return profiler_ns, module


def extract_slots_from_file(cpp_file: Path) -> List[SlotUsage]:
    try:
        content = cpp_file.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  ⚠️  Cannot read {cpp_file}: {e}")
        return []

    patterns = [
        r"MORI_TRACE_SPAN\s*\([^,]+,\s*Slot::(\w+)",
        r"MORI_TRACE_NEXT\s*\([^,]+,\s*Slot::(\w+)",
        r"MORI_TRACE_INSTANT\s*\([^,]+,\s*Slot::(\w+)",
    ]

    usages = []
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            slot_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1
            usages.append(
                SlotUsage(
                    slot_name=slot_name, source_file=cpp_file, line_number=line_num
                )
            )

    return usages


def make_enum_name(base_name: str) -> str:
    parts = base_name.split("_")
    return "".join(p.capitalize() for p in parts) + "Slot"


def group_slots_by_file(
    usages: List[SlotUsage], project_root: Path
) -> Dict[str, SlotGroup]:
    groups = defaultdict(lambda: {"slots": set(), "files": set()})

    for usage in usages:
        base_name = usage.source_file.stem
        groups[base_name]["slots"].add(usage.slot_name)
        groups[base_name]["files"].add(usage.source_file)

    result = {}
    for base_name, data in groups.items():
        if not data["slots"]:
            continue

        representative_file = sorted(data["files"])[0]
        explicit_ns = extract_profiler_namespace_hint(representative_file)
        profiler_ns, module_path = infer_profiler_namespace(
            representative_file, project_root
        )
        final_ns = explicit_ns if explicit_ns else profiler_ns
        enum_name = make_enum_name(base_name)
        sorted_slots = sorted(data["slots"])

        result[base_name] = SlotGroup(
            base_name=base_name,
            module_path=module_path,
            profiler_namespace=final_ns,
            enum_name=enum_name,
            slots=sorted_slots,
            source_files=sorted(data["files"]),
        )

    return result


def print_summary(groups: Dict[str, SlotGroup]):
    print(f"\n{'='*70}")
    print("Generation Summary")
    print(f"{'='*70}")

    total_slots = sum(len(g.slots) for g in groups.values())

    print(f"\n  Total slot groups: {len(groups)}")
    print(f"  Total unique slots: {total_slots}")
    print("\n  Groups by module:")

    by_module = defaultdict(list)
    for group in groups.values():
        by_module[group.module_path].append(group)

    for module, module_groups in sorted(by_module.items()):
        print(f"\n    {module}/")
        for group in sorted(module_groups, key=lambda g: g.base_name):
            print(f"      • {group.enum_name}: {len(group.slots)} slot(s)")
            print(f"        namespace: {group.profiler_namespace}")
            print(f"        python: mori.cpp.{group.enum_name}s")


def write_if_changed(path: Path, content: str) -> bool:
    if path.exists():
        try:
            current_content = path.read_text(encoding="utf-8")
            if current_content == content:
                return False
        except (IOError, OSError):
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: generate_profiler_bindings.py <project_root> <source_dir> <output_include_dir> <output_pybind_file>"
        )
        sys.exit(1)

    project_root = Path(sys.argv[1]).resolve()
    source_dir = Path(sys.argv[2]).resolve()
    output_include_dir = Path(sys.argv[3]).resolve()
    output_pybind_file = Path(sys.argv[4]).resolve()

    print(f"{'='*70}")
    print("Profiler Code Generator")
    print(f"{'='*70}")
    print(f"\n  Scanning: {source_dir}")

    all_usages = []
    scanned_files = 0
    for cpp_file in source_dir.rglob("*.cpp"):
        scanned_files += 1
        usages = extract_slots_from_file(cpp_file)
        if usages:
            unique_slots = len(set(u.slot_name for u in usages))
            print(f"  ✓ {cpp_file.relative_to(source_dir)}: {unique_slots} slot(s)")
            all_usages.extend(usages)

    print(f"\n  Scanned {scanned_files} file(s), found {len(all_usages)} slot usage(s)")

    if not all_usages:
        print("\n  No profiler slots detected. Generating empty bindings.")
        written = write_if_changed(
            output_pybind_file, "// No profiler slots detected\n"
        )
        if written:
            print(f"  ✓ Wrote {output_pybind_file}")
        else:
            print(f"  - Skipped {output_pybind_file} (unchanged)")
        return

    print(f"\n{'='*70}")
    print("Generating Code")
    print(f"{'='*70}\n")

    # Check for collisions before grouping
    file_base_map = defaultdict(list)
    for usage in all_usages:
        base_name = usage.source_file.stem
        if usage.source_file not in file_base_map[base_name]:
            file_base_map[base_name].append(usage.source_file)

    collisions = {k: v for k, v in file_base_map.items() if len(v) > 1}
    if collisions:
        print("\n  ❌ Error: Filename collisions detected!")
        print("     The profiler generation uses the filename (stem) as the unique ID.")
        print("     The following filenames appear in multiple locations:")
        for name, files in collisions.items():
            print(f"\n     • {name}:")
            for f in files:
                print(f"       - {f.relative_to(project_root)}")
        print(
            "\n     Please rename the files to be unique or update the generation script."
        )
        sys.exit(1)

    groups = group_slots_by_file(all_usages, project_root)

    print("Generating headers:")
    for base_name, group in sorted(groups.items()):
        # generate_header inlined logic to use write_if_changed
        xmacro_name = f"{group.base_name.upper()}_PROFILER_SLOTS"
        xmacro_lines = []
        for slot in group.slots:
            py_name = camel_to_snake(slot)
            xmacro_lines.append(f'  X({slot}, "{py_name}")')

        xmacro_def = f"#define {xmacro_name}(X) \\\n" + " \\\n".join(xmacro_lines)

        max_slots = len(group.slots)
        static_assert = (
            f"static_assert(static_cast<int>({group.enum_name}::MAX_SLOTS) == {max_slots}, "
            f'"Slot count mismatch in {group.enum_name}");'
        )

        macro_prefix = f"{group.base_name.upper()}_PROFILER"

        header = f"""// AUTO-GENERATED FILE - DO NOT EDIT

#pragma once

{xmacro_def}

namespace {group.profiler_namespace} {{

enum class {group.enum_name} : int {{
#define X(name, str) name,
  {xmacro_name}(X)
#undef X
  MAX_SLOTS
}};

{static_assert}

}}  // namespace {group.profiler_namespace}

#ifdef ENABLE_PROFILER
#define {macro_prefix}_SLOT_TYPE {group.profiler_namespace}::{group.enum_name}
#define {macro_prefix}_INIT_CONTEXT(name, cfg, gwid, lid) \\
  MORI_DECLARE_PROFILER_CONTEXT(name, {macro_prefix}_SLOT_TYPE, \\
                                mori::core::profiler::ProfilerContext, \\
                                mori::core::profiler::ProfilerContext(cfg, gwid, lid))
#else
#define {macro_prefix}_SLOT_TYPE int
#define {macro_prefix}_INIT_CONTEXT(name, cfg, gwid, lid) ((void)0)
#endif
"""

        module_dir = output_include_dir / group.module_path
        output_file = module_dir / f"{group.base_name}_slots.hpp"
        if write_if_changed(output_file, header):
            print(f"  ✓ {output_file.relative_to(output_include_dir.parent.parent)}")
        else:
            print(
                f"  - {output_file.relative_to(output_include_dir.parent.parent)} (unchanged)"
            )

    # Unified header
    includes = []
    for base_name, group in sorted(groups.items()):
        includes.append(
            f'#include "mori/profiler/{group.module_path}/{base_name}_slots.hpp"'
        )

    header = f"""// AUTO-GENERATED FILE - DO NOT EDIT

#pragma once

#ifdef ENABLE_PROFILER

{chr(10).join(includes)}

#endif  // ENABLE_PROFILER
"""
    unified_file = output_include_dir / "profiler.hpp"

    if write_if_changed(unified_file, header):
        try:
            rel_path = unified_file.relative_to(output_include_dir.parent.parent)
            print(f"\n  ✓ Unified header: {rel_path}")
        except ValueError:
            print(f"\n  ✓ Unified header: {unified_file}")
    else:
        print("\n  - Unified header (unchanged)")

    print("\nGenerating Python bindings:")
    # generate_pybind_cpp inlined logic to use write_if_changed
    binding_functions = []
    registrations = []

    for base_name, group in sorted(groups.items()):
        header_path = f"mori/profiler/{group.module_path}/{base_name}_slots.hpp"
        xmacro_name = f"{base_name.upper()}_PROFILER_SLOTS"
        py_group_name = group.enum_name + "s"

        binding_func = f"""
void Bind{py_group_name}(pybind11::module_& m) {{
  std::vector<std::pair<const char*, int>> slots;
  int counter = 0;
#define X(name, str) slots.push_back({{str, counter++}});
  {xmacro_name}(X)
#undef X
  mori::pybind::BindProfilerSlots(m, "{py_group_name}", slots);
}}
"""
        binding_functions.append((header_path, binding_func))
        registrations.append(f"MORI_REGISTER_PROFILER_SLOTS(Bind{py_group_name});")

    includes = sorted(set(h for h, _ in binding_functions))
    functions = [f for _, f in binding_functions]

    # Check for slot ID collisions across modules
    slot_id_map = {}  # {id: [module_names]}
    for base_name, group in sorted(groups.items()):
        for idx in range(len(group.slots)):
            if idx not in slot_id_map:
                slot_id_map[idx] = []
            slot_id_map[idx].append(base_name)

    has_collision = any(len(modules) > 1 for modules in slot_id_map.values())

    # Generate ALL_PROFILER_SLOTS merger function
    all_slots_function = """
void BindAllProfilerSlots(pybind11::module_& m) {
  pybind11::dict all_slots;
"""

    if has_collision:
        all_slots_function += """
  // WARNING: Multiple modules detected. Using prefixed names to avoid collisions.
  // If multiple modules share the same trace buffer, consider using separate buffers
  // or manually allocating non-overlapping slot ID ranges.
"""
        for base_name, group in sorted(groups.items()):
            xmacro_name = f"{base_name.upper()}_PROFILER_SLOTS"
            prefix = base_name
            all_slots_function += f"""
  // {group.enum_name} (prefixed as "{prefix}.*")
  {{
    int counter = 0;
#define X(name, str) all_slots[pybind11::int_(counter++)] = pybind11::str("{prefix}." str);
    {xmacro_name}(X)
#undef X
  }}
"""
    else:
        # No collision, use simple names
        for base_name, group in sorted(groups.items()):
            xmacro_name = f"{base_name.upper()}_PROFILER_SLOTS"
            all_slots_function += f"""
  // Merge {group.enum_name}
  {{
    int counter = 0;
#define X(name, str) all_slots[pybind11::int_(counter++)] = pybind11::str(str);
    {xmacro_name}(X)
#undef X
  }}
"""

    all_slots_function += """
  m.attr("ALL_PROFILER_SLOTS") = all_slots;
}
"""

    cpp_content = f"""// AUTO-GENERATED FILE - DO NOT EDIT

#include "mori/pybind/profiler_registry.hpp"
#include <pybind11/pybind11.h>

{chr(10).join(f'#include "{inc}"' for inc in includes)}

namespace {{

{chr(10).join(functions)}

{all_slots_function}

{chr(10).join(registrations)}

MORI_REGISTER_PROFILER_SLOTS(BindAllProfilerSlots);

}}  // namespace
"""
    if write_if_changed(output_pybind_file, cpp_content):
        try:
            rel_path = output_pybind_file.relative_to(project_root)
            print(f"  ✓ Python bindings: {rel_path}")
        except ValueError:
            print(f"  ✓ Python bindings: {output_pybind_file}")
    else:
        print("  - Python bindings (unchanged)")

    print_summary(groups)

    print(f"\n{'='*70}")
    print("Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
