export default class SRINodeNormFailure extends Error {

  constructor(message = 'SRI Resolver Failed.', ...params) {
    super(...params);

    Object.setPrototypeOf(this, SRINodeNormFailure.prototype);

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, SRINodeNormFailure);
    }

    this.name = 'SRINodeNormFailure';
    this.message = message;
  }
}
