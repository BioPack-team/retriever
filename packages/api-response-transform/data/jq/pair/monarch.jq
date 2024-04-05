generateCurie($edge.input.id; $edge.input.curies) as $input_curie 
| if $edge.response_mapping[$edge.predicate].output_name | contains("object")
  then "object_namespace"
  else "subject_namespace"
  end 
| . as $output_namespace
| { $input_curie: [$response] } | .[$input_curie][0].items = [
  $response.items[]
  | select(. != null)
  # | select(
  #   .[$output_namespace] == $edge.output.id
  # ) 
  | .publications = [
    if .publications != null
    then .publications
    else []
    end
    | .[]
    | select(. | startswith("PMID:"))
  ]
  | .primary_knowledge_source as $primary_knowledge_source
  | .sources = (
    if .primary_knowledge_source != null and .aggregator_knowledge_source != null
    then
      [
        {
          resource_role: "primary_knowledge_source",
          resource_id: $primary_knowledge_source
        }
      ] 
      + (.aggregator_knowledge_source | reverse | reduce .[] as $source (
        null; 
        if . == null
        then [{
          resource_role: "aggregator_knowledge_source", 
          resource_id: $source,
          upstream_resource_ids: [$primary_knowledge_source]
        }]
        else . + [{
          resource_role: "aggregator_knowledge_source",
          resource_id: $source,
          upstream_resource_ids: [.[-1].resource_id]
        }]
        end
      ))
    else [{ resource_role: "primary_knowledge_source", resource_id:"infores:monarchinitiative" }]
    end
  )
]
