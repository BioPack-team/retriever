if $edge.query.method == "post" then
  # if response is not an array, then use response.hits
  if (.response | type) == "array" then .response else .response.hits end |
  reduce .[] as $item ({};
    # if the item is notfound, then proceed to next item & keep current object
    if ($item | keys | contains(["notfound"])) then
      .
    else
      generateCurieWithInputs(
        $edge.input.id;
        $item.query; 
        $edge.input.curies
        | toArray
        | map(. | split(":") | last)
      ) as $curie | .[$curie] += [$item]
    end
  )
else
  .response as $res | generateCurie(
    $edge.input.id;
    ($edge.input.curies | toArray | map(. | split(":") | last) | first)
  ) as $curie | {} | .[$curie] = [$res]
end
