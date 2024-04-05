export default class MetaKGLoadingError extends Error {
  statusCode: number;
  constructor(message = "Failed to load metakg", ...params: string[]) {
    super(...params);

    this.name = "MetaKGLoadingError";
    this.message = message;
    this.statusCode = 400;
  }
}
