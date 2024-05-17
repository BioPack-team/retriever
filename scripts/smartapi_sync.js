process.env.DEBUG="retriever*"
process.env.SYNC_AND_EXIT="true" 
const sync = require("../packages/server/built/controllers/cron/update_local_smartapi.js").default;
sync();
