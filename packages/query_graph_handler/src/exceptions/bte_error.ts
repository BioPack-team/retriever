export default class BTEError extends Error {
  statusCode: string;
  constructor(message = 'Query aborted', name = 'QueryAborted', code = '501', ...params: any) {
    super(...params);

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, BTEError);
    }

    this.name = name;
    this.message = message;
    this.statusCode = code;
  }
}

module.exports = BTEError;
