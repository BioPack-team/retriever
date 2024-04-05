import QueryGraphHelper from '../../src/helper';

describe('Test helper moduler', () => {
  const nodeObject1 = {
    getID() {
      return 'n01';
    },
    getCategory() {
      return 'Node1Type';
    },
  };
  const nodeObject2 = {
    getID() {
      return 'n02';
    },
    getCategory() {
      return 'Node2Type';
    },
  };

  test('Test _generateHash function', () => {
    const res = QueryGraphHelper._generateHash('123');
    expect(res.length).toBe(32);
    const res1 = QueryGraphHelper._generateHash('kkkkkkkkk');
    expect(res1.length).toBe(32);
  });
});
