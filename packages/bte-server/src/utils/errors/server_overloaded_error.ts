export default class ServerOverloadedError extends Error {
  status: number;
  retryAfter: number;
  constructor(message: string, retryAfter: number, ...params: string[]) {
    super(...params);
    this.name = "ServerOverloadedError";
    this.message = message ?? "Server is overloaded, please try again later.";
    this.status = 503;
    this.retryAfter = retryAfter ?? 60;
  }
}
