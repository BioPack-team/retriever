import smartapiCron from "./update_local_smartapi";
import cacheClearCron from "./clear_edge_cache";

export default function scheduleCronJobs() {
  smartapiCron();
  cacheClearCron();
}
