#!/usr/bin/env bash

/bin/rm -rf .beacon
termframe -W 130 -H auto --mode dark -o public/generated/beacon_setup.svg -- beacon setup

beacon index --yes
termframe -W 130 -H auto --mode dark -o public/generated/beacon_index.svg -- beacon index
termframe -W 130 -H 50 --mode dark -o public/generated/beacon_index.svg -- expect -c '
    set timeout 90
    spawn beacon index

    expect -timeout 60 "Parsing"
    expect -timeout 60 "100%"
    expect -timeout 60 "Loading jina-code-embeddings"
    expect -timeout 60 "FTS5 index"
    expect -timeout 60 "Index complete"

    after 600
    expect eof
  '


termframe -W 130 -H auto --mode dark -o public/generated/beacon_search.svg -- beacon search "embedding function with transformers"

# simulate interactive query
termframe -W 130 -H 50 --mode dark -o public/generated/beacon_ask.svg -- expect -c '
    set timeout 90
    spawn beacon ask
    expect "search your codebase"
    send "where do i do semantic embedding\r"
    expect -timeout 60 "navigate"
    after 600
    send "\033\[B"
    after 400
    send "q"
    expect eof
  '
# fix the title bar text from command
python -c "
import re, html
from pathlib import Path
svg = Path('public/generated/beacon_ask.svg')
svg.write_text(re.sub(
    r'(<line\b[^/]*/><text\b[^>]*>).*?(</text>)',
    lambda m: m.group(1) + html.escape('beacon ask') + m.group(2),
    svg.read_text(), count=1, flags=re.DOTALL
))
"

