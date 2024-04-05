export default class FailToLoadSpecError extends Error {
  constructor(message = "Loading spec failed") {
    super(message);
    Object.setPrototypeOf(this, FailToLoadSpecError.prototype);
    this.name = "FailedToLoadSpecError";
    this.message = message;
  }
}
