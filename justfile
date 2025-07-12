#!/usr/bin/env -S just --justfile

[group: 'dev']
install:
  maturin build --release; python3 -m pip install .

[group: 'dev']
test:
  python3 -m pytest python/tests/

[group: 'dev']
docs:
  python3 docs/parse.py

[group: 'bench']
runtime:
  #!/bin/bash
  README="python/benches/README.md"

  if [ -f $README ]; then
    rm $README
  fi

  touch $README
  echo "# benches" >> $README
  echo "" >> $README
  python3 python/benches/bench_polygon.py >> $README


[group: 'check']
clippy:
  cargo clippy --all --all-targets --all-features -- --deny warnings
