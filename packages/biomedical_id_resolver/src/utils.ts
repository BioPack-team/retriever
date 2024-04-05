import { ObjectWithStringKeyAndArrayValues } from './common/types';
import { CURIE, APIMETA } from './config';

export function generateCurie(prefix: string, val: string | number): string {
  if (CURIE.ALWAYS_PREFIXED.includes(prefix)) {
    return val.toString();
  } else {
    return prefix + ':' + val.toString();
  }
}

export function getPrefixFromCurie(curie: string): string {
  return curie.split(':')[0];
}

export function generateDBID(val: string): string {
  if (!CURIE.ALWAYS_PREFIXED.includes(val.split(':')[0])) {
    return val.substring(val.indexOf(':') + 1);
  } else {
    return val;
  }
}

export function appendArrayOrNonArrayObjectToArray(lst: any[], item: any) {
  if (Array.isArray(item)) {
    for (const val of item) {
      if (typeof val === 'string') {
        lst.push(val);
      } else if (typeof val === 'number') {
        lst.push(val.toString());
      }
    }
    return lst;
  } else {
    if (typeof item === 'string') {
      lst.push(item);
    } else if (typeof item === 'number') {
      lst.push(item.toString());
    }
    return lst;
  }
}

export function generateObjectWithNoDuplicateElementsInValue(
  input: ObjectWithStringKeyAndArrayValues,
): ObjectWithStringKeyAndArrayValues {
  Object.keys(input).map((key) => {
    input[key] = Array.from(new Set(input[key]));
  });
  return input;
}

export function generateIDTypeDict(): ObjectWithStringKeyAndArrayValues {
  const res = {} as ObjectWithStringKeyAndArrayValues;
  Object.values(APIMETA).map((metadata) => {
    for (const prefix of metadata.id_ranks) {
      if (!(prefix in res)) {
        res[prefix] = [];
      }
      res[prefix].push(metadata.semantic);
    }
  });
  return res;
}
