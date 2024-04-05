import QEdge from '../../src/query_edge';
import QNode from '../../src/query_node';

function basicQEdge({
  predicates,
  subjectIds,
  objectIds,
  reverse = false,
}: { predicates?: string[]; subjectIds?: string[]; objectIds?: string[]; reverse?: boolean } = {}): QEdge {
  return new QEdge({
    id: 'e01',
    predicates,
    subject: new QNode({
      id: 'n01',
      ids: subjectIds,
    }),
    object: new QNode({
      id: 'n02',
      ids: objectIds,
    }),
  });
}

describe('Test QEdge class', () => {
  describe('Test getPredicate function', () => {
    test('Non reversed edge should return predicates itself', () => {
      const edge = basicQEdge({
        predicates: ['biolink:treats'],
      });
      const res = edge.getPredicate();
      expect(res).toContain('treats');
    });

    test('Undefined predicate should return itself', () => {
      const edge = basicQEdge();
      const res = edge.getPredicate();
      expect(res).toBeUndefined;
    });

    test('An array of non-undefined predicates should return itself', () => {
      const edge = basicQEdge({
        predicates: ['biolink:treats', 'biolink:targets'],
      });
      const res = edge.getPredicate();
      expect(res).toContain('treats');
      expect(res).toContain('targets');
    });

    test('An array of non-undefined predicates with reverse edge should exclude return value if undefined', () => {
      const edge = basicQEdge({
        predicates: ['biolink:treats', 'biolink:targets'],
        objectIds: ['yes'],
      });
      const res = edge.getPredicate();
      expect(res).toContain('treated_by');
    });

    test('An array of non-undefined predicates with reverse edge should return reversed predicates if not undefined', () => {
      const edge = basicQEdge({
        predicates: ['biolink:treats', 'biolink:targets'],
        objectIds: ['yes'],
      });
      const res = edge.getPredicate();
      expect(res).toContain('treated_by');
    });
  });

  describe('Test getOutputNode function', () => {
    test('reversed edge should return the subject', () => {
      const edge = basicQEdge({
        predicates: ['biolink:treats', 'biolink:targets'],
        objectIds: ['yes'],
      });
      const res = edge.getOutputNode();
      expect(res.id).toEqual('n01');
    });

    test('non reversed edge should return the object', () => {
      const edge = basicQEdge({
        predicates: ['biolink:treats', 'biolink:targets'],
        subjectIds: ['yes'],
      });
      const res = edge.getOutputNode();
      expect(res.id).toEqual('n02');
    });
  });
});
