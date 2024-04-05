type FIELDS_FOR_FILTER =
  | "input_type"
  | "output_type"
  | "predicate"
  | "api_name"
  | "source"
  | "component"
  | "qualifiers";

export const FILTER_FIELDS: FIELDS_FOR_FILTER[] = [
  "input_type",
  "output_type",
  "predicate",
  "api_name",
  "source",
  "component",
  "qualifiers",
];

export const SMARTAPI_URL =
  "https://smart-api.info/api/query?q=tags.name:translator&size=1000&fields=paths,servers,tags,components.x-bte*,info,_meta";

export const SINGLE_API_SMARTAPI_QUERY_TEMPLATE = "https://smart-api.info/api/metadata/{smartapi_id}";

export const TEAM_SMARTAPI_QUERY_TEMPLATE =
  'https://smart-api.info/api/query?q=info.x-translator.team:"{team_name}"&size=1000&fields=paths,servers,tags,components.x-bte*,info,_meta';

export const COMPONENT_SMARTAPI_QUERY_TEMPLATE =
  'https://smart-api.info/api/query?q=info.x-translator.component:"{component_name}"&size=1000&fields=paths,servers,tags,components.x-bte*,info,_meta';

export const TAG_SMARTAPI_QUERY_TEMPLATE =
  "https://smart-api.info/api/query?q=tags.name:{tag_name}&size=1000&fields=paths,servers,tags,components.x-bte*,info,_meta";
