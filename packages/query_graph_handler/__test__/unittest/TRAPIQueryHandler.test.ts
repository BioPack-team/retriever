import { Record } from '@biothings-explorer/api-response-transform';
import path from 'path';
import fs from 'fs';
import _ from 'lodash';
const AxiosActual = jest.requireActual('axios');
import InferredQueryHandler from '../../src/inferred_mode/inferred_mode';
import TRAPIQueryHandler, { InvalidQueryGraphError } from '../../src/index';
import axios from 'axios';

jest.mock('../../src/inferred_mode/inferred_mode');
jest.mock('axios');

const records = Record.unfreezeRecords(
  JSON.parse(fs.readFileSync(path.resolve(__dirname, '../data/queryRecords.json'), { encoding: 'utf8' })),
);

describe('test TRAPIQueryHandler methods', () => {
  const OLD_ENV = process.env;

  beforeEach(() => {
    jest.resetAllMocks();
    jest.resetModules();
    process.env = { ...OLD_ENV };
  });

  afterEach(() => {
    jest.resetAllMocks();
    process.env = OLD_ENV;
  });

  test('processQueryGraph', async () => {
    const handler = new TRAPIQueryHandler();
    const invalidQueryGraph = {
      nodes: {},
      edges: {},
    };
    try {
      await handler._processQueryGraph(invalidQueryGraph);
    } catch (err) {
      expect(err).toBeInstanceOf(InvalidQueryGraphError);
    }
  });

  test.skip('dumpRecords', async () => {
    let fakeFile = '';
    jest.mock('fs', () => ({
      ...(jest.requireActual('fs') as typeof fs),
      promises: {
        ...(jest.requireActual('fs').promises as typeof fs.promises),
        writeFile: jest.fn().mockImplementation((path, str) => {
          fakeFile += str;
        }),
      },
    }));
    const handler = new TRAPIQueryHandler();

    process.env.DUMP_RECORDS = 'fileLocation';

    // await handler.dumpRecords(records);
    expect(fs.promises.writeFile).toHaveBeenCalled();
    expect(fakeFile.length).toBeGreaterThan(0);
    const allHashes = records.map((record) => record.recordHash);
    Record.unfreezeRecords(JSON.parse(fakeFile)).forEach((record) => {
      expect(allHashes).toContain(record.recordHash);
    });
  });

  test('queryIsOneHop', () => {
    const query = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, '../data/chemicalEntity_treats_acanthosis.json'), { encoding: 'utf8' }),
    );
    const handler = new TRAPIQueryHandler();
    handler.setQueryGraph(query.message.query_graph);

    expect(handler._queryIsOneHop).toBeTruthy();
  });

  test('_handleInferredEdges', async () => {
    const query = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, '../data/chemicalEntity_treats_acanthosis.json'), { encoding: 'utf8' }),
    );
    (InferredQueryHandler.prototype.query as jest.Mock).mockReturnValueOnce('fakeResponse');
    const handler = new TRAPIQueryHandler();
    handler.setQueryGraph(query.message.query_graph);
    await handler._handleInferredEdges();
    expect(handler.getResponse()).toEqual('fakeResponse');
  });

  test('query', async () => {
    jest.mock('axios');
    (axios.post as jest.Mock).mockResolvedValue({ data: { 'WIKIPATHWAYS:WP195': null } });

    const query = JSON.parse(
      fs.readFileSync(path.resolve(__dirname, '../data/chemicals_targeting_IL1_Signaling_Pathway.json'), {
        encoding: 'utf8',
      }),
    );
    const handler1 = new TRAPIQueryHandler(
      {
        smartAPIID: 'fakeID',
      },
      undefined,
      undefined,
      false,
    );
    handler1.setQueryGraph(query.message.query_graph);
    await handler1.query();
    expect(handler1.logs.map((log) => log.level)).toContain('ERROR');
    const handler2 = new TRAPIQueryHandler(
      {
        teamName: 'fakeID',
      },
      undefined,
      undefined,
      false,
    );
    handler2.setQueryGraph(query.message.query_graph);
    await handler2.query();
    expect(handler2.logs.map((log) => log.level)).toContain('ERROR');
    const handler3 = new TRAPIQueryHandler({
      smartAPIID: '59dce17363dce279d389100834e43648',
    });
    handler3.setQueryGraph(query.message.query_graph);
    await handler3.query();
    expect(handler3.logs[handler3.logs.length - 1].message).toMatch(
      'smartAPI/team-specific endpoints only support single-edge queries',
    );
    const twoHopInferred = _.cloneDeep(query);
    twoHopInferred.message.query_graph.edges.e01.knowledge_type = 'inferred';
    const handler4 = new TRAPIQueryHandler();
    handler4.setQueryGraph(twoHopInferred.message.query_graph);
    await handler4.query();
    expect(handler4.logs[handler4.logs.length - 1].message).toMatch(
      'Inferred Mode edges are only supported in single-edge queries',
    );
  }, 10000);

  test('findUnregisteredAPI', async () => {
    const handler = new TRAPIQueryHandler({
      apiList: {
        include: [
          {
            id: 'fakeID',
            name: 'fake API',
          },
        ],
        exclude: []
      },
    });
    const unregisteredAPIs = await handler.findUnregisteredAPIs();
    expect(unregisteredAPIs).toContain('fakeID');
  });
});
