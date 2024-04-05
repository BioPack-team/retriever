import TRAPIQueryHandler from '../../src/index';
import fs from 'fs';
import path from 'path';

describe('Testing TRAPIQueryHandler Module', () => {
  const example_folder = path.resolve(__dirname, '../data');

  //skip until we figure out why it returns no results
  //https://suwulab.slack.com/archives/CC218TEKC/p1624558136437200
  test.skip('When looking for chemicals affected by Phenotype Increased Urinary Glycerol, Glycerol should pop up', async () => {
    const queryHandler = new TRAPIQueryHandler({}, undefined, undefined, true);
    const query = JSON.parse(
      fs.readFileSync(path.join(example_folder, 'increased_urinary_glycerol_affects_glycerol.json'), {
        encoding: 'utf8',
      }),
    );
    queryHandler.setQueryGraph(query.message.query_graph);
    await queryHandler.query();
    const res = queryHandler.getResponse();
    expect(res.message.knowledge_graph.nodes).toHaveProperty('CHEBI:17754');
  });

  // skip until integrity tests can be rewritten with mocked API responses
  test.skip('When looking for genes related to Disease DYSKINESIA, FAMILIAL, WITH FACIAL MYOKYMIA, ACDY5 should pop up', async () => {
    const queryHandler = new TRAPIQueryHandler({}, undefined, undefined, true);
    const query = JSON.parse(
      fs.readFileSync(path.join(example_folder, 'FDFM_caused_by_ACDY5.json'), { encoding: 'utf8' }),
    );
    queryHandler.setQueryGraph(query.message.query_graph);
    await queryHandler.query();
    const res = queryHandler.getResponse();
    // console.log(res);
    expect(res.message.knowledge_graph.nodes).toHaveProperty('NCBIGene:111');
  });

  //skip this test for now as the test query needs to be re-evaluated and the value of 'CHEBI:3962' needs to be updated.
  test.skip('When looking for chemicals targeting IL1 Signaling patway, curcumin should pop up', async () => {
    const queryHandler = new TRAPIQueryHandler({}, undefined, undefined, true);
    const query = JSON.parse(
      fs.readFileSync(path.join(example_folder, 'chemicals_targeting_IL1_Signaling_Pathway.json'), {
        encoding: 'utf8',
      }),
    );
    queryHandler.setQueryGraph(query.message.query_graph);
    await queryHandler.query();
    const res = queryHandler.getResponse();
    expect(res.message.knowledge_graph.nodes).toHaveProperty('CHEBI:3962');
  });
});
