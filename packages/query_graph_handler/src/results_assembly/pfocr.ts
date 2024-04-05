import axios from 'axios';
import Debug from 'debug';
const debug = Debug('bte:biothings-explorer-trapi:pfocr');
import { intersection } from '../utils';
import _ from 'lodash';
import { LogEntry, StampedLog } from '@biothings-explorer/utils';
import { TrapiResult } from '../types';

// the minimum acceptable intersection size between the CURIEs
// in a TRAPI result and in a PFOCR figure.
const MATCH_COUNT_MIN = 2;
const FIGURE_COUNT_MAX = 20;

interface pfocrQueryBody {
  q: string[];
  scopes: string;
  fields: string[];
  operator: string;
  analyzer: string;
  minimum_should_match: number;
  size: number;
  with_total: boolean;
  from?: number;
}

interface FigureResult {
  _id: string;
  notfound?: boolean;
  associatedWith: {
    figureUrl: string;
    pmc: string;
    mentions: {
      genes: {
        ncbigene: string[];
      };
    };
  };
}

interface RawFigureResult extends FigureResult {
  query: string;
}

interface DeDupedFigureResult extends FigureResult {
  query: Set<string>;
}

/* Get all results by using a scrolling query
 * https://docs.mygene.info/en/latest/doc/query_service.html#scrolling-queries
 * Using new POST scrolling behavior here:
 * https://github.com/newgene/biothings.api/pull/52
 * All queries must include size: 1000, with_total: true
 * Subsequent queries must use `from` to designate a starting point
 */
async function getAllByScrolling(
  baseUrl: string,
  queryBody: pfocrQueryBody,
  batchIndex: number,
  hits: RawFigureResult[] = [],
): Promise<RawFigureResult[]> {
  queryBody.from = batchIndex;
  const { data } = await axios.post(baseUrl, queryBody).catch((err) => {
    debug('Error in scrolling request', err);
    throw err;
  });

  hits.push(...data.hits);
  debug(`Batch window ${batchIndex}-${batchIndex + 1000}: ${data.hits.length} hits retrieved for PFOCR figure data`);
  if (batchIndex + 1000 < data.max_total) {
    return await getAllByScrolling(baseUrl, queryBody, batchIndex + 1000, hits);
  } else {
    return hits;
  }
}

/* qTerms are the CURIEs that go with the 'q' query parameter.
 */
async function getPfocrFigures(qTerms: Set<string>): Promise<DeDupedFigureResult[]> {
  debug(`Getting PFOCR figure data`);
  const url = 'https://biothings.ncats.io/pfocr/query';
  /*
   * We can now POST using minimum_should_match to bypass most set logic on our side
   * detailed here: https://github.com/biothings/pending.api/issues/88
   */

  const figureResults: RawFigureResult[] = [];
  await Promise.all(
    _.chunk([...qTerms], 1000).map(async (qTermBatch) => {
      const queryBody = {
        q: [...qTermBatch],
        scopes: 'associatedWith.mentions.genes.ncbigene', // TODO better system when we use more than NCBIGene
        fields: ['_id', 'associatedWith.mentions.genes.ncbigene', 'associatedWith.pmc', 'associatedWith.figureUrl'],
        operator: 'OR',
        analyzer: 'whitespace',
        minimum_should_match: MATCH_COUNT_MIN,
        size: 1000,
        with_total: true,
      };

      figureResults.push(...(await getAllByScrolling(url, queryBody, 0)));
    }),
  ).catch((err) => {
    debug('Error getting PFOCR figures (getPfocrFigures)', err);
    throw err;
  });

  /*
   * When we make separate queries for different CURIEs, we can get
   * duplicate figures, so we need to de-duplicate.
   */
  const mergedFigureResults: { [figureID: string]: DeDupedFigureResult } = {};
  const figuresAdded = new Set();
  figureResults.map((figure) => {
    const figureId = figure._id;
    if (!figure.notfound && !figuresAdded.has(figureId)) {
      figuresAdded.add(figureId);
      mergedFigureResults[figureId] = { ...figure, query: new Set([figure.query]) };
    } else if (!figure.notfound && figuresAdded.has(figureId)) {
      mergedFigureResults[figureId].query.add(figure.query);
    }
  });

  debug(`${Object.values(mergedFigureResults).length} total PFOCR figure hits retrieved`);
  return Object.values(mergedFigureResults);
}

function getMatchableQNodeIDs(allTrapiResults: TrapiResult[]): Set<string> {
  const matchableQNodeIDs: Set<string> = new Set();

  if (allTrapiResults.length === 0) {
    return matchableQNodeIDs;
  }

  // TODO: this will need to be updated to handle non-NCBIGene CURIEs as well
  // as non-gene CURIEs once we support querying for chemicals and diseases.

  const supportedPrefixes = new Set(['NCBIGene']);
  for (const trapiResult of allTrapiResults) {
    for (const [qNodeID, nodeBindingValues] of Object.entries(trapiResult.node_bindings)) {
      for (const nodeBindingValue of nodeBindingValues) {
        const prefix = nodeBindingValue.id.split(':')[0];
        if (supportedPrefixes.has(prefix)) {
          matchableQNodeIDs.add(qNodeID);
          break;
        }
      }
    }
  }

  debug(`QNode(s) having CURIEs that PFOCR could potentially match: ${[...matchableQNodeIDs]}`);
  return matchableQNodeIDs;
}

/* time complexity: O(t*f)
 * where
 * t: trapiResults.length
 * f: figures.length
 */
export async function enrichTrapiResultsWithPfocrFigures(allTrapiResults: TrapiResult[]): Promise<StampedLog[]> {
  const matchableQNodeIDs = getMatchableQNodeIDs(allTrapiResults);
  const logs: StampedLog[] = [];
  let resultsWithTruncatedFigures = 0;
  const truncatedFigures: Set<string> = new Set();

  if (matchableQNodeIDs.size < MATCH_COUNT_MIN) {
    // No TRAPI result can satisfy MATCH_COUNT_MIN
    logs.push(new LogEntry('DEBUG', null, 'Query does not match criteria, skipping PFOCR figure enrichment.').getLog());
    return logs;
  }

  // TODO: currently just NCBIGene CURIEs. Expand to handle any CURIE in PFOCR.

  const trapiResultToCurieSet: Map<TrapiResult, string> = new Map();

  const curieCombinations: Set<string> = new Set(
    allTrapiResults.reduce((arr: string[], res) => {
      const resultCuries: Set<string> = new Set();
      const matchedQNodes: Set<string> = new Set();
      [...matchableQNodeIDs].forEach((qNodeID) => {
        res.node_bindings[qNodeID]
          .map((node_binding) => node_binding.id)
          .filter((curie) => curie.startsWith('NCBIGene:'))
          .forEach((curie) => {
            resultCuries.add(curie);
            matchedQNodes.add(qNodeID);
          });
      });

      const resultCuriesString = [...resultCuries].map((curie) => curie.replace('NCBIGene:', '')).join(' ');

      if (resultCuries.size >= MATCH_COUNT_MIN && matchedQNodes.size >= MATCH_COUNT_MIN) {
        trapiResultToCurieSet.set(res, resultCuriesString);
        arr.push(resultCuriesString);
      }

      return arr;
    }, []),
  );

  const figures = await getPfocrFigures(curieCombinations).catch((err) => {
    debug('Error getting PFOCR figures (enrichTrapiResultsWithPfocrFigures)', err);
    throw err;
  });

  debug(`${figures.length} PFOCR figures match at least ${MATCH_COUNT_MIN} genes from any TRAPI result`);

  const figuresByCuries: { [queryCuries: string]: DeDupedFigureResult[] } = {};
  figures.forEach((figure) => {
    [...figure.query].forEach((queryCuries) => {
      figuresByCuries[queryCuries] =
        queryCuries in figuresByCuries ? [...figuresByCuries[queryCuries], figure] : [figure];
    });
  });

  const matchedFigures: Set<string> = new Set();
  const matchedTrapiResults: Set<TrapiResult> = new Set();

  // const allGenesInAllFigures = figures.reduce((set, fig) => {
  //   fig.associatedWith.mentions.genes.ncbigene.forEach((gene) => set.add(gene));
  //   return set;
  // }, new Set() as Set<string>);

  for (const trapiResult of allTrapiResults) {
    // No figures match this result
    if (!figuresByCuries[trapiResultToCurieSet.get(trapiResult)]) continue;

    const resultCuries: Set<string> = new Set();
    const resultMatchableQNodeIDs: Set<string> = new Set();
    [...matchableQNodeIDs].forEach((qNodeID) => {
      trapiResult.node_bindings[qNodeID]
        .map((node_binding) => node_binding.id)
        .filter((curie) => curie.startsWith('NCBIGene:'))
        .forEach((curie) => {
          resultCuries.add(curie.replace('NCBIGene:', ''));
          resultMatchableQNodeIDs.add(qNodeID);
        });
    });
    if (resultMatchableQNodeIDs.size < 2) continue;

    (figuresByCuries[trapiResultToCurieSet.get(trapiResult)] ?? []).forEach((figure) => {
      if (!('pfocr' in trapiResult)) {
        trapiResult.pfocr = [];
      }

      const figureCurieSet = new Set(figure.associatedWith.mentions.genes.ncbigene);
      const resultGenesInFigure = intersection(resultCuries, figureCurieSet);

      const matchedQNodes = [...matchableQNodeIDs].filter((matchableQNodeID) => {
        const currentQNodeCurieSet = new Set(
          trapiResult.node_bindings[matchableQNodeID].map((node_binding) => node_binding.id),
        );

        return (
          intersection(currentQNodeCurieSet, new Set([...resultGenesInFigure].map((geneID) => `NCBIGene:${geneID}`)))
            .size > 0
        );
      });

      // If we've matched on 2 curies, but we haven't actually matched on multiple nodes
      if (matchedQNodes.length < 2) return;

      const otherGenesInFigure = figureCurieSet.size - resultGenesInFigure.size;

      const resultGenesInOtherFigures = [...resultCuries].filter((gene) => {
        return figures.some((fig) => fig.associatedWith.mentions.genes.ncbigene.includes(gene));
      }).length;
      // let otherGenesInOtherFigures = [...allGenesInAllFigures].filter((gene) => {
      //   return !resultCuries.has(gene) && !figureCurieSet.has(gene);
      // }).length;

      const precision = resultGenesInFigure.size / (resultGenesInFigure.size + otherGenesInFigure);
      const recall = resultGenesInFigure.size / (resultGenesInFigure.size + resultGenesInOtherFigures);

      trapiResult.pfocr.push({
        figureUrl: figure.associatedWith.figureUrl,
        pmc: figure.associatedWith.pmc,
        // TODO: do we want to include figure title? Note: this would need to be added to queryBody.
        //title: figure.associatedWith.title,
        matchedCuries: [...resultGenesInFigure].map((geneID) => `NCBIGene:${geneID}`),
        score: 2 * ((precision * recall) / (precision + recall)),
        // 1 -
        // parseFloat(
        //   Analyze([
        //     [resultGenesInFigure.size, resultGenesInOtherFigures],
        //     [otherGenesInFigure, otherGenesInOtherFigures],
        //   ]).pValue,
        // ),
      });
      matchedTrapiResults.add(trapiResult);
    });

    // Sort by score and cut down to top 20
    const sortedFigures = trapiResult.pfocr.sort((figA, figB) => {
      return figB.score - figA.score;
    });

    if (sortedFigures.length > FIGURE_COUNT_MAX) {
      resultsWithTruncatedFigures += 1;
      sortedFigures.slice(0, 20).forEach((figure) => truncatedFigures.add(figure.figureUrl));
      // debug(`Truncating ${sortedFigures.length} PFOCR figures to ${FIGURE_COUNT_MAX} for TRAPI result w/ curies ${trapiResultToCurieSet.get(trapiResult).split(' ').map((ID) => `NCBIGene:${ID}`).join(', ')}`)
    }

    trapiResult.pfocr = sortedFigures.slice(0, 20);
    trapiResult.pfocr.map((figure) => matchedFigures.add(figure.figureUrl));
  }

  // Each of the matched figures has at least one TRAPI result with an overlap of 2+ genes.
  // Each of the matched TRAPI results has at least one figure with an overlap of 2+ genes.
  const unusedFigures = [...truncatedFigures].filter((figureUrl) => !matchedFigures.has(figureUrl)).length;
  const message = `${resultsWithTruncatedFigures} results had pfocr figures truncated to max of 20 (${truncatedFigures.size} unique figures removed, ${unusedFigures} not appearing elsewhere in results).`;
  debug(message);
  logs.push(new LogEntry('DEBUG', null, message).getLog());
  debug(
    `${MATCH_COUNT_MIN}+ CURIE matches: ${matchedFigures.size} PFOCR figures and ${matchedTrapiResults.size} TRAPI results`,
  );
  logs.push(
    new LogEntry(
      'INFO',
      null,
      `${matchedTrapiResults.size} results successfully enriched with ${matchedFigures.size} unique PFOCR figures.`,
    ).getLog(),
  );

  return logs;
}

module.exports.enrichTrapiResultsWithPfocrFigures = enrichTrapiResultsWithPfocrFigures;
