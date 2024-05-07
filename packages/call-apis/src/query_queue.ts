import RateCounter from "./rate_limiter";
import { redisClient } from "@retriever/utils";
import Debug from "debug";
import Subquery from "./queries/subquery";
import { QueryHandlerOptions } from "@retriever/types";
const debug = Debug("retriever:call-apis:query");

export interface SubqueryBundle {
  query: Subquery;
  options: QueryHandlerOptions;
}

export default class APIQueryQueue {
  queue: SubqueryBundle[];
  rateCounter: RateCounter;
  constructor() {
    this.queue = [];
    this.rateCounter = new RateCounter(redisClient);
  }

  get length() {
    return this.queue.length;
  }

  get isEmpty() {
    return this.queue.length === 0;
  }

  add(query: Subquery, options: QueryHandlerOptions) {
    this.queue.unshift({ query, options });
  }

  async getNext(): Promise<{ query: Subquery; options: QueryHandlerOptions }> {
    const next = this.queue.pop();
    if (!next) return;
    const { query, options } = next;
    const queryDelayed = query.delayUntil && query.delayUntil >= new Date();
    if ((await this.rateCounter.atLimit(query)) || queryDelayed) {
      debug(
        [
          `query to ${query.APIEdge.query_operation.server}`,
          `rate-limited or delayed, will-retry after rest of sub-query queue`,
        ].join(" "),
      );
      this.queue.unshift({ query, options });
      return new Promise(resolve => {
        setImmediate(async () => {
          resolve(await this.getNext());
        });
      });
    }
    return { query, options };
  }
}
