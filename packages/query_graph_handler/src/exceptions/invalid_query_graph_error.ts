export default class InvalidQueryGraphError extends Error {
  statusCode: number;
  constructor(message = 'Your Input Query Graph is invalid.', ...params: string[]) {
    super(...params);

    Object.setPrototypeOf(this, InvalidQueryGraphError.prototype);

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, InvalidQueryGraphError);
    }

    this.name = 'InvalidQueryGraphError';
    this.message = message;
    this.statusCode = 400;
  }
}
