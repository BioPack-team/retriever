export default class WorkflowError extends Error {
  statusCode: number;
  constructor(message = "BTE doesn't handle the operations specified in the workflow field", ...params: string[]) {
    super(...params);
    this.name = "WorkflowError";
    this.message = message;
    this.statusCode = 400;
  }
}

