#!/bin/sh
# Run the board-level self-test (sim/tb_board_top.v) against deploy/rtl/board_top.v.
#
#   sh sim/run_board_sim.sh
#
# Everything is resolved relative to this script, so the tree can live anywhere.
# Vivado is located via $VIVADO_BIN, then a list of common install paths.
#
# Why the unisims: board_top instantiates MMCME2_BASE and BUFG, which need the
# Xilinx primitive models compiled in.  MMCME2_BASE itself instantiates
# MMCME2_ADV, so all three must be present or elaboration fails with
# "Module <MMCME2_ADV> not found".  glbl is required as a second top so that the
# global set/reset net is defined.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

# xvlog only accepts Windows-style paths (F:/...), not MSYS paths (/f/...)
win() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
HERE_W=$(win "$HERE")
ROOT_W=$(win "$ROOT")

VBIN=""
for cand in \
    "${VIVADO_BIN:-}" \
    "/f/vivado/2025.2/Vivado/bin" \
    "/c/Xilinx/Vivado/2025.2/bin" \
    "/c/Xilinx/Vivado/2024.2/bin" \
    "/c/Xilinx/Vivado/2023.2/bin" \
    "/d/Xilinx/Vivado/2025.2/bin" \
    "/e/Xilinx/Vivado/2025.2/bin" ; do
    if [ -n "$cand" ] && [ -f "$cand/xvlog.bat" ]; then VBIN="$cand"; break; fi
done
if [ -z "$VBIN" ]; then
    echo "ERROR: Vivado not found. Set VIVADO_BIN to its bin directory, e.g."
    echo "  export VIVADO_BIN=/c/Xilinx/Vivado/2025.2/bin"
    exit 1
fi
VBIN_W=$(win "$VBIN")
XSRC_W=$(win "$(dirname "$VBIN")/../data/verilog/src")

echo "Vivado : $VBIN_W"
echo "root   : $ROOT_W"
echo

# Vivado batch mode needs this explicitly or loader.bat exits 1 with no output
export PROCESSOR_ARCHITECTURE=AMD64

BUILD="$HERE/board_sim"
mkdir -p "$BUILD"
cd "$BUILD" || exit 1

# $readmemh in board_top.v resolves ".mem" against the CWD, so the data must be here
for f in feat_all.mem exp_dec.mem w2d.mem b2d.mem th2d.mem fc_w2d.mem fc_b2d.mem; do
    cp -f "$ROOT/deploy/data/$f" . || { echo "ERROR: missing deploy/data/$f"; exit 1; }
done

rm -rf xsim.dir

"$VBIN_W/xvlog.bat" --nolog \
    "$XSRC_W/glbl.v" \
    "$XSRC_W/unisims/MMCME2_ADV.v" \
    "$XSRC_W/unisims/MMCME2_BASE.v" \
    "$XSRC_W/unisims/BUFG.v" \
    "$ROOT_W/rtl/conv2d_engine.v" \
    "$ROOT_W/rtl/rate_encoder2d.v" \
    "$ROOT_W/rtl/snn2d_top.v" \
    "$ROOT_W/deploy/rtl/board_top.v" \
    "$ROOT_W/sim/tb_board_top.v" > xvlog.log 2>&1
rc=$?
echo "xvlog=$rc"
[ $rc -ne 0 ] && { echo "--- xvlog.log ---"; cat xvlog.log; exit 1; }

"$VBIN_W/xelab.bat" --nolog -debug typical tb_board_top glbl -s tb_board > xelab.log 2>&1
rc=$?
echo "xelab=$rc"
[ $rc -ne 0 ] && { echo "--- xelab.log ---"; tail -20 xelab.log; exit 1; }

"$VBIN_W/xsim.bat" --nolog tb_board -runall > xsim.log 2>&1
# xsim often returns non-zero after a clean $finish; judge by the log instead
echo "xsim finished (exit $?)"

echo
echo "===== result ====="
grep -aE "===|cases done|decisions seen|LED|pass_count|done_all|FAIL|TIMEOUT" xsim.log
