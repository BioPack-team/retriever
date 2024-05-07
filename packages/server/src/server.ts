import Debug from "debug";
const debug = Debug("retriever:server-start");

async function main() {
  await testRedisConnection(); // must happen before app to avoid issues
  const { default: app } = await import("./app");
  const { default: cron } = await import("./controllers/cron/index");
  const PORT = Number.parseInt(process.env.PORT) || 3000;
  cron();
  process.env.DEBUG_COLORS = "true";
  app.listen(PORT, () => {
    debug(`Instance Env: ${process.env.INSTANCE_ENV ?? "local"}`);
    console.log(`⭐⭐⭐ BioThings Explorer is ready! ⭐ Try it now @ http://localhost:${PORT} ✨`);
  });
}

async function testRedisConnection() {
  const { redisClient } = await import("@retriever/utils");

  if (redisClient.clientEnabled) {
    // redis enabled
    debug("Checking connection to redis...");
    try {
      await redisClient.client.pingTimeout();
      debug("Redis connection successful.");
    } catch (error) {
      debug(`Redis connection failed due to error ${error}`);
      debug(`Disabling redis for current server runtime...`);
      process.env.INTERNAL_DISABLE_REDIS = "true";
    }
  }
}

main();
