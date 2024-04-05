export default class PredicatesLoadingError extends Error {
  statusCode: number;
  constructor(message = "Failed to load metakg", ...params: string[]) {
    super(...params);

    this.name = "PredicatesLoadingError";
    this.message = message;
    this.statusCode = 400;
  }
}
