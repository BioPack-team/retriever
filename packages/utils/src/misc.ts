export function toArray<Type>(input: Type | Type[]): Type[] {
  if (Array.isArray(input)) {
    return input;
  }
  return [input];
}

export function getUnique<Type>(input: Type[]): Type[] {
  return Array.from(new Set(input));
}

export function removeBioLinkPrefix(input: string): string {
  if (input && input.startsWith('biolink:')) {
    return input.slice(8);
  }
  return input;
}

// This gets the intersection of two sets.
// Lodash _.intersection gets the intersection of two arrays.
// see https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Set
export function intersection<Type>(setA: Set<Type>, setB: Set<Type>): Set<Type> {
  const resultSet: Set<Type> = new Set();
  for (const elem of setB) {
    if (setA.has(elem)) {
      resultSet.add(elem);
    }
  }
  return resultSet;
}

// see https://stackoverflow.com/a/29585704
export function cartesian(a: number[][]): number[][] {
  // a = array of arrays
  let i: number, j: number, l: number, m: number;
  const o = [];
  if (!a || a.length == 0) return a;

  const a1 = a.splice(0, 1)[0]; // the first array of a
  a = cartesian(a);
  for (i = 0, l = a1.length; i < l; i++) {
    if (a && a.length) {
      for (j = 0, m = a.length; j < m; j++) {
        o.push([a1[i]].concat(a[j]));
      }
    } else {
      o.push([a1[i]]);
    }
  }
  return o;
}
