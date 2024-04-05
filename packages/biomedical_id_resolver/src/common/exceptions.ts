export = class IrresolvableIDResolverInputError extends Error {
  constructor(message: string) {
    super(message);

    // Set the prototype explicitly.
    Object.setPrototypeOf(this, IrresolvableIDResolverInputError.prototype);
  }
};
