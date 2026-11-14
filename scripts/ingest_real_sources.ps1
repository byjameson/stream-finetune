# Rebuild the verified knowledge DB used by the local assistant.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$db = "$root/data/knowledge_real.db"

$sources = @(
  @{u="https://docs.python.org/3/tutorial/introduction.html"; t="Python tutorial: introduction"},
  @{u="https://docs.python.org/3/tutorial/datastructures.html"; t="Python tutorial: data structures"},
  @{u="https://docs.python.org/3/tutorial/controlflow.html"; t="Python tutorial: control flow"},
  @{u="https://docs.python.org/3/library/math.html"; t="Python math library"},
  @{u="https://numpy.org/doc/stable/user/absolute_beginners.html"; t="NumPy absolute beginners"}
)
foreach ($s in $sources) {
  python -m streamft spiking knowledge --db $db add --url $s.u --title $s.t --trust official | Out-Null
}
python -m streamft spiking knowledge --db $db add `
  --text "Project deployment note: the local assistant runs on a 6 GB RTX 3050 laptop and uses verified retrieval." `
  --title "Operator note" --source "human:operator" --trust human | Out-Null
python -m streamft spiking knowledge --db $db add `
  --text "The moon is made of cheese." `
  --title "False moon claim" --source "https://random.example/moon" --trust untrusted | Out-Null
python -m streamft spiking knowledge --db $db verify --threshold 0.5 --min-support 2
python -m streamft spiking knowledge --db $db review --id 6 --approve --note "operator note" --reviewer operator
python -m streamft spiking knowledge --db $db review --id 7 --reject --note "clearly false" --reviewer operator
python -m streamft assistant index --db $db
