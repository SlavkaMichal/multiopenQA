#!/bin/bash
#
# NOTES
#
# - Must expand templates to avoid a large loss of content.
# - Text will not (redundantly) contain the title string.
# - Keep sections. Section title will be marked by "Section::::".
# - Keep lists. List bullets will be marked by "BULLET::::".
# - Keep tables. They're mostly garbage but can be removed later (remove "^!*").
# - Remove disambiguation pages. Right now there is no use for them.

WIKILANG=$1
if [[ "$WIKILANG" == "" ]]; then
  WIKILANG=$1
fi

if [[ "$WIKILANG" == "" ]]; then
  echo "WIKILANG not set"
  exit 1
fi

echo $WIKILANG

INPUT=data/wiki/$WIKILANG/*.xml
PROCESSES=12
OUTPUT=data/wiki/$WIKILANG/json
#TEMPLATES=

echo "Input: ${INPUT}"
echo "Processes: ${PROCESSES}"
echo "Output: ${OUTPUT}"

python scripts/WikiExtractor.py $INPUT \
       --json \
       --processes $PROCESSES \
       --output $OUTPUT \
       --bytes 1M \
       --sections \
       --min_text_length 0 \
       --filter_disambig_pages
#       --templates $TEMPLATES \
