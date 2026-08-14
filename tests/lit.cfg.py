import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst


config.name = "GraphForge"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]
config.test_source_root = os.path.join(config.graphforge_src_root, "tests")
config.test_exec_root = os.path.join(config.graphforge_obj_root, "tests")

llvm_config.use_default_substitutions()
tool_dirs = [config.graphforge_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    [ToolSubst("gf-opt"), ToolSubst("gf-translate"),
     ToolSubst("FileCheck"), ToolSubst("not")], tool_dirs
)
