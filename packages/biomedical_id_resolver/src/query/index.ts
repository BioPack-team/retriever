import { Scheduler } from './scheduler';
import { DBIdsObject, IndividualResolverOutput } from '../common/types';
import Debug from 'debug';
const debug = Debug('biomedical-id-resolver:Query');

const query = async (resolvable: DBIdsObject): Promise<IndividualResolverOutput[]> => {
  const scheduler = new Scheduler(resolvable);
  scheduler.schedule();
  let result = [];
  for (const promises of Object.values(scheduler.buckets)) {
    const res = (await Promise.allSettled(promises)) as any;
    res.map((item) => {
      if (item.status === 'fulfilled') {
        result.push(item.value);
      } else {
        debug(`One API Query fails, reason is ${item.reason}`);
      }
    });
  }
  return result;
};
export default query;
