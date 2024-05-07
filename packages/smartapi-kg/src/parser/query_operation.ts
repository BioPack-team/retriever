import {
  QueryOperationInterface,
  XBTEKGSOperationObject,
  XBTEParametersObject,
} from "./types";

export default class QueryOperationObject implements QueryOperationInterface {
  params: XBTEParametersObject;
  requestBody: any;
  requestBodyType: string;
  supportBatch: boolean;
  batchSize: number;
  useTemplating: boolean;
  inputSeparator: string;
  path: string;
  method: string;
  server: string;
  tags: string[];
  pathParams: string[];
  templateInputs: any;

  set xBTEKGSOperation(newOp: XBTEKGSOperationObject) {
    this.params = newOp.parameters;
    this.requestBody = newOp.requestBody;
    this.requestBodyType = newOp.requestBodyType;
    this.supportBatch = newOp.supportBatch;
    this.useTemplating = newOp.useTemplating;
    this.inputSeparator = newOp.inputSeparator;
    this.templateInputs = newOp.templateInputs;
    this.batchSize = newOp.batchSize;
  }
}
