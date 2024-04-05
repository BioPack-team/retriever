import axios from 'axios';
import axiosRetry from 'axios-retry';
import { CURIE } from './config';
import { SRIResolverOutput, ResolverInput, SRIBioEntity, SRIResponseEntity, SRIResponse } from './common/types';
import Debug from 'debug';
import _ from 'lodash';
import SRINodeNormFailure from './exceptions/sri_resolver_failiure';
const debug = Debug('bte:biomedical-id-resolver:SRI');

/** sets up request retry policy (weird typescript b/c old axios version) */
// @ts-ignore
axiosRetry(axios, {
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (err) => {
    return axiosRetry.isNetworkOrIdempotentRequestError(err) || err.response?.status >= 500;
  },
});

/** convert object of arrays into array of unique IDs */
function combineInputs(userInput: ResolverInput): string[] {
  const result = Object.keys(userInput).reduce(function (r, k) {
    return r.concat(userInput[k]);
  }, []);
  return [...new Set(result)];
}

/**
 * input: array of curies
 * handles querying and batching of inputs
 */
async function query(api_input: string[]) {
  const url = {
    dev: 'https://nodenormalization-sri.renci.org/get_normalized_nodes',
    ci: 'https://nodenorm.ci.transltr.io/get_normalized_nodes',
    test: 'https://nodenorm.test.transltr.io/get_normalized_nodes',
    prod: 'https://nodenorm.transltr.io/get_normalized_nodes',
  }[process.env.INSTANCE_ENV ?? 'prod'];

  const chunked_input = _.chunk(api_input, 1000);
  try {
    const userAgent = [
      `BTE/${process.env.NODE_ENV === 'production' ? 'prod' : 'dev'}`,
      `Node/${process.version} ${process.platform}`,
    ].join(' ');
    const axios_queries = chunked_input.map((input) => {
      return axios.post(
        url,
        { curies: input, conflate: true, drug_chemical_conflate: true },
        { headers: { 'User-Agent': userAgent } },
      );
    });
    //convert res array into single object with all curies
    let res = await Promise.all(axios_queries);
    res = res.map((r, i) => {
      return Object.keys(r.data).length ? r.data : Object.fromEntries(chunked_input[i].map((curie) => [curie, null]));
    });
    return Object.assign({}, ...res);
  } catch (err) {
    throw new SRINodeNormFailure(`SRI resolver failed: ${err.message}`);
  }
}

/** Build id resolution object for curies that couldn't be resolved */
function UnresolvableEntry(curie: string, semanticType: string): SRIBioEntity {
  return {
    primaryID: curie,
    equivalentIDs: [curie],
    label: curie,
    labelAliases: [curie],
    primaryTypes: [semanticType],
    semanticTypes: [semanticType],
    attributes: {},
  };
}

/** Build id resolution object for curies that were successfully resolved by SRI */
function ResolvableEntry(SRIEntry: SRIResponseEntity): SRIBioEntity {
  // Temporary fix for https://github.com/biothings/biothings_explorer/issues/652
  if (SRIEntry.type?.[0].includes('NamedThing')) SRIEntry.type.reverse();
  return {
    primaryID: SRIEntry.id.identifier,
    equivalentIDs: [
      ...SRIEntry.equivalent_identifiers.reduce((set, { identifier }) => {
        if (identifier) set.add(identifier);
        return set;
      }, new Set<string>()),
    ],
    label: SRIEntry.id.label || SRIEntry.id.identifier,
    labelAliases: [
      ...SRIEntry.equivalent_identifiers.reduce((set, { label }) => {
        if (label) set.add(label);
        return set;
      }, new Set<string>()),
    ],
    // first semantic type is the "primary" type
    primaryTypes: [SRIEntry.type?.[0].split(':')[1]],
    semanticTypes: SRIEntry.type,
    attributes: {},
  };
}

/** Transform output from SRI into original resolver shape */
function transformResults(results: SRIResponse): SRIResolverOutput {
  return Object.fromEntries(
    Object.entries(results).map(([curie, entity]) => {
      return [curie, entity ? ResolvableEntry(entity) : UnresolvableEntry(curie, null)];
    }),
  );
}

/** add entries with original semantic types if they don't match the SRI resolved types */
function mapInputSemanticTypes(originalInput: ResolverInput, result: SRIResolverOutput): SRIResolverOutput {
  Object.entries(originalInput).forEach(([semanticType, originalCuries]) => {
    if (['unknown', 'undefined', 'NamedThing'].includes(semanticType)) return;
    const uniqueInputs = [...new Set(originalCuries)];
    uniqueInputs.forEach((curie) => {
      const entry = result[curie];
      const primaryTypesMissing = [!entry.primaryTypes, !entry.primaryTypes[0], entry.primaryTypes.length === 0];
      if (primaryTypesMissing.some((condition) => condition)) {
        entry.primaryTypes = [semanticType];
        entry.semanticTypes = [semanticType];
      } else if (!entry.primaryTypes.includes(semanticType)) {
        debug(
          [
            `SRI resolved type '${entry.primaryTypes[0]}'`,
            `doesn't match input semantic type '${semanticType}'`,
            `for curie '${entry.primaryID}'. Adding entry for '${semanticType}'.`,
          ].join(' '),
        );
        entry.primaryTypes.push(semanticType);
      }
    });
  });
  return result;
}

export async function _resolveSRI(userInput: ResolverInput): Promise<SRIResolverOutput> {
  const uniqueInputIDs = combineInputs(userInput);

  let queryResults = await query(uniqueInputIDs);

  queryResults = transformResults(queryResults);

  queryResults = mapInputSemanticTypes(userInput, queryResults);

  return queryResults;
}
