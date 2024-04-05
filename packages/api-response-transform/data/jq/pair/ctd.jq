(reduce .response[].Input as $id (
  {}; if (. as $mapping | $id | in($mapping) | not)
  then
    ($edge.input.curies | find((. | ascii_upcase) == ($id | split(":") | last | ascii_upcase))) as $fixedID
    | .[$id] = $fixedID
  end
)) as $idMapping # maps response[].Input -> appropriate $edge.input.curies
| reduce .response[] as $item (
  {}; if ($idMapping[$item.Input])
  then
    (generateCurie($edge.input.id; $idMapping[$item.Input])) as $curie
    | .[$curie] += [$item]
  end
) | map_values([.])
