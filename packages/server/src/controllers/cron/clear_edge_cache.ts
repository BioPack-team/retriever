import { redisClient } from "@biothings-explorer/utils";
import Debug from "debug";
const debug = Debug("bte:biothings-explorer-trapi:cron");
import cron from "node-cron";

export default function scheduleClearCache() {
  cron.schedule("0 0 * * *", async () => {
    debug("Checking status for edge cache clearing.");
    if (!redisClient.clientEnabled) {
      debug("Cache not enabled, skipping edge cache clearing");
      return;
    }
    debug("Redis client enabled, proceeding with cache clearing.");
    await redisClient.client.usingLock(["redisLock:EdgeCaching"], 600000, async () => {
      redisClient.client.clearEdgeCache();
    });
  });
}
