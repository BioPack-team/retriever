import {
  SmartAPIPathItemObject,
  ParsedAPIMetadataObject,
  SmartAPIKGOperationObject,
  SmartAPIOperationObject,
  XBTEKGSOperationObject,
  XBTEKGSOperationBioEntityObject,
  SmartAPIReferenceObject,
} from "./types";
import QueryOperationObject from "./query_operation";

export default class Endpoint {
  pathItemObject: SmartAPIPathItemObject;
  apiMetadata: ParsedAPIMetadataObject;
  path: string;

  constructor(pathItemObject: SmartAPIPathItemObject, apiMetadata: ParsedAPIMetadataObject, path: string) {
    this.pathItemObject = pathItemObject;
    this.apiMetadata = apiMetadata;
    this.path = path;
  }

  private fetchPathParams(OperationObject: SmartAPIOperationObject): string[] {
    const params: string[] = [];
    if (!("parameters" in OperationObject)) {
      return params;
    }
    for (const param of OperationObject.parameters) {
      if (param.in === "path") {
        params.push(param.name);
      }
    }
    return params;
  }

  private constructQueryOperation({
    op,
    method,
    pathParams,
  }: {
    op: XBTEKGSOperationObject;
    method: string;
    pathParams: string[];
  }) {
    const server = this.apiMetadata.url;
    const queryOperation = new QueryOperationObject();
    queryOperation.xBTEKGSOperation = op;
    queryOperation.method = method;
    queryOperation.pathParams = pathParams;
    queryOperation.server = server;
    queryOperation.path = this.path;
    queryOperation.tags = this.apiMetadata.tags;
    return queryOperation;
  }

  private removeBioLinkPrefix(input: string | undefined) {
    if (typeof input === "undefined") {
      return input;
    }
    if (input.startsWith("biolink:")) {
      return input.replace(/biolink:/gi, "");
    }
    return input;
  }

  private resolveRefIfProvided(rec: SmartAPIReferenceObject) {
    return "$ref" in rec ? this.apiMetadata.components.fetchComponentByRef(rec.$ref) : rec;
  }

  private constructAssociation(
    input: XBTEKGSOperationBioEntityObject,
    output: XBTEKGSOperationBioEntityObject,
    op: XBTEKGSOperationObject,
  ) {
    return {
      input_id: this.removeBioLinkPrefix(input.id),
      input_type: this.removeBioLinkPrefix(input.semantic),
      output_id: this.removeBioLinkPrefix(output.id),
      output_type: this.removeBioLinkPrefix(output.semantic),
      predicate: this.removeBioLinkPrefix(op.predicate),
      qualifiers: op.qualifiers
        ? Object.fromEntries(
            Object.entries(op.qualifiers).map(([qualifierType, qualifier]) => [
              this.removeBioLinkPrefix(qualifierType),
              qualifier,
            ]),
          )
        : undefined,
      source: op.source,
      knowledge_level: op.knowledge_level,
      agent_type: op.agent_type,
      api_name: this.apiMetadata.title,
      smartapi: this.apiMetadata.smartapi,
      "x-translator": this.apiMetadata["x-translator"],
    };
  }

  private constructResponseMapping(op: XBTEKGSOperationObject) {
    if ("responseMapping" in op) {
      op.response_mapping = op.responseMapping;
    }
    return {
      [op.predicate]: this.resolveRefIfProvided(op.response_mapping),
    };
  }

  private parseIndividualOperation({
    op,
    method,
    pathParams,
  }: {
    op: XBTEKGSOperationObject;
    method: string;
    pathParams: string[];
  }) {
    const res = [];
    const queryOperation = this.constructQueryOperation({
      op,
      method,
      pathParams,
    });
    const responseMapping = this.constructResponseMapping(op);
    for (const input of op.inputs) {
      for (const output of op.outputs) {
        let updateInfo = {} as SmartAPIKGOperationObject;
        const association = this.constructAssociation(input, output, op);
        updateInfo = {
          query_operation: queryOperation,
          association,
          response_mapping: responseMapping,
          tags: queryOperation.tags,
        };
        res.push(updateInfo);
      }
    }
    return res;
  }

  constructEndpointInfo(): SmartAPIKGOperationObject[] {
    let res = [] as SmartAPIKGOperationObject[];
    ["get", "post"].map(method => {
      if (method in this.pathItemObject) {
        const pathParams = this.fetchPathParams(this.pathItemObject[method]);
        if (
          "x-bte-kgs-operations" in this.pathItemObject[method] &&
          Array.isArray(this.pathItemObject[method]["x-bte-kgs-operations"])
        ) {
          let operation;
          let op;
          for (const rec of this.pathItemObject[method]["x-bte-kgs-operations"]) {
            operation = this.resolveRefIfProvided(rec);
            operation = Array.isArray(operation) ? operation : [operation];
            for (op of operation) {
              res = [...res, ...this.parseIndividualOperation({ op, method, pathParams })];
            }
          }
        }
      }
    });
    return res;
  }
}
