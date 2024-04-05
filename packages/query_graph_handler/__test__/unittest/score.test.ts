import { calculateScore, exportForTesting } from '../../src/results_assembly/score';
const { record_weight, text_mined_record_weight, ngd_weight, LENGTH_PENALTY, scaled_sigmoid } = exportForTesting;

describe('Test score function', () => {
  const ngdPairs = {
    'C0678941-C0267841': 0.5,
    'C4548369-C0678941': 0.6,
    'C4548369-C0267841': 0.7,
  };

  const sampleComboSimple = [
    {
      inputQNodeID: 'nB',
      outputQNodeID: 'nC',
      inputPrimaryCuries: new Set(['UMLS:C0678941']),
      outputPrimaryCuries: new Set(['MONDO:0006633']),
      inputUMLS: new Set(['C0678941']),
      outputUMLS: new Set(['C0267841']),
      isTextMined: [true],
      qEdgeID: 'eB',
      recordHashes: new Set(['a']),
    },
    {
      inputQNodeID: 'nA',
      outputQNodeID: 'nB',
      inputPrimaryCuries: new Set(['PUBCHEM.COMPOUND:77843966']),
      outputPrimaryCuries: new Set(['UMLS:C0678941']),
      inputUMLS: new Set(['C4548369']),
      outputUMLS: new Set(['C0678941']),
      isTextMined: [true],
      qEdgeID: 'eA',
      recordHashes: new Set(['b']),
    },
  ];

  const sampleComboComplex = [
    {
      inputQNodeID: 'nB',
      outputQNodeID: 'nC',
      inputPrimaryCuries: new Set(['UMLS:C0678941']),
      outputPrimaryCuries: new Set(['MONDO:0006633']),
      inputUMLS: new Set(['C0678941']),
      outputUMLS: new Set(['C0267841']),
      isTextMined: [true, false, true],
      qEdgeID: 'eB',
      recordHashes: new Set(['a', 'b', 'c']),
    },
    {
      inputQNodeID: 'nA',
      outputQNodeID: 'nB',
      inputPrimaryCuries: new Set(['PUBCHEM.COMPOUND:77843966']),
      outputPrimaryCuries: new Set(['UMLS:C0678941']),
      inputUMLS: new Set(['C4548369']),
      outputUMLS: new Set(['C0678941']),
      isTextMined: [true, true, true],
      qEdgeID: 'eA',
      recordHashes: new Set(['b', 'c', 'd']),
    },
    {
      inputQNodeID: 'nA',
      outputQNodeID: 'nC',
      inputPrimaryCuries: new Set(['PUBCHEM.COMPOUND:77843966']),
      outputPrimaryCuries: new Set(['MONDO:0006633']),
      inputUMLS: new Set(['C4548369']),
      outputUMLS: new Set(['C0267841']),
      isTextMined: [false, false],
      qEdgeID: 'eC',
      recordHashes: new Set(['c', 'd']),
    },
  ];

  test('Test calculateScore function - simple case w/ ngd', () => {
    const eAScore = text_mined_record_weight + ngd_weight * (1 / ngdPairs['C4548369-C0678941']);
    const eBScore = text_mined_record_weight + ngd_weight * (1 / ngdPairs['C0678941-C0267841']);
    const expected_score = scaled_sigmoid((eBScore + eAScore) / Math.pow(2, LENGTH_PENALTY));

    const res = calculateScore(sampleComboSimple, ngdPairs);
    expect(res.score).toBe(expected_score);
    expect(res.scoredByNGD).toBeTruthy();
  });

  test('Test calculateScore function - simple case w/o ngd', () => {
    const eAScore = text_mined_record_weight;
    const eBScore = text_mined_record_weight;
    const expected_score = scaled_sigmoid((eBScore + eAScore) / Math.pow(2, LENGTH_PENALTY));

    const res = calculateScore(sampleComboSimple, {});
    expect(res.score).toBe(expected_score);
    expect(res.scoredByNGD).toBeFalsy();
  });

  test('Test calculateScore function - complex case w/ ngd', () => {
    const eAScore = 2 * text_mined_record_weight + 1 * record_weight + ngd_weight * (1 / ngdPairs['C4548369-C0678941']);
    const eBScore = 3 * text_mined_record_weight + 0 * record_weight + ngd_weight * (1 / ngdPairs['C0678941-C0267841']);
    const eCScore = 0 * text_mined_record_weight + 2 * record_weight + ngd_weight * (1 / ngdPairs['C4548369-C0267841']);

    const expected_score = scaled_sigmoid(
      (eBScore + eAScore) / Math.pow(2, LENGTH_PENALTY) + eCScore / Math.pow(1, LENGTH_PENALTY),
    );

    const res = calculateScore(sampleComboComplex, ngdPairs);
    expect(res.score).toBe(expected_score);
    expect(res.scoredByNGD).toBeTruthy();
  });

  test('Test calculateScore function - complex case w/o ngd', () => {
    const eAScore = 2 * text_mined_record_weight + 1 * record_weight;
    const eBScore = 3 * text_mined_record_weight + 0 * record_weight;
    const eCScore = 0 * text_mined_record_weight + 2 * record_weight;

    const expected_score = scaled_sigmoid(
      (eBScore + eAScore) / Math.pow(2, LENGTH_PENALTY) + eCScore / Math.pow(1, LENGTH_PENALTY),
    );

    const res = calculateScore(sampleComboComplex, {});
    expect(res.score).toBe(expected_score);
    expect(res.scoredByNGD).toBeFalsy();
  });
});
