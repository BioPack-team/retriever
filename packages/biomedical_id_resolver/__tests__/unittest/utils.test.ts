import { generateCurie, getPrefixFromCurie, generateObjectWithNoDuplicateElementsInValue, appendArrayOrNonArrayObjectToArray, generateDBID, generateIDTypeDict } from '../../src/utils';


describe("Test Utils Module", () => {
    describe("Test generateCurie function", () => {
        test("id that is always a curie should return itself", () => {
            const inputID = 'MONDO:000123';
            const res = generateCurie('MONDO', inputID);
            expect(res).toEqual(inputID);
        })

        test("id that is not always a curie should return prefix plus colon and itself", () => {
            const inputID = '1017';
            const res = generateCurie('NCBIGene', inputID);
            expect(res).toEqual('NCBIGene:1017');
        })

        test("id that is Irresolvable should return prefix plus colon and itself", () => {
            const inputID = '1017';
            const res = generateCurie('NCBIGene1', inputID);
            expect(res).toEqual('NCBIGene1:1017');
        })
    })

    describe("Test getPrefixFromCurie function", () => {
        test("prefix part is parsed if it's a curie", () => {
            const inputID = 'MONDO:000123';
            const res = getPrefixFromCurie(inputID);
            expect(res).toEqual('MONDO');
        })

        test("return itself if input is not a curie", () => {
            const inputID = '1017';
            const res = getPrefixFromCurie(inputID);
            expect(res).toEqual(inputID);
        })
    })

    describe("Test appendArrayOrNonArrayObjectToArray function", () => {
        test("append as a whole if input is not an array", () => {
            const input = 'MONDO:000123';
            const lst = ['123']
            const res = appendArrayOrNonArrayObjectToArray(lst, input);
            expect(res).toHaveLength(2);
            expect(res).toEqual(['123', 'MONDO:000123'])
        })

        test("append each item if input is an array", () => {
            const input = ['MONDO:000123', 'MONDO:000124'];
            const lst = ['123']
            const res = appendArrayOrNonArrayObjectToArray(lst, input);
            expect(res).toHaveLength(3);
            expect(res).toEqual(['123', 'MONDO:000123', 'MONDO:000124'])
        })

        test("convert element of input to string if any of the element in input is number", () => {
            const input = ['MONDO:000123', 1017, 1018.1];
            const lst = ['123']
            const res = appendArrayOrNonArrayObjectToArray(lst, input);
            expect(res).toHaveLength(4);
            expect(res).toEqual(['123', 'MONDO:000123', '1017', '1018.1'])
        })

        test("convert item to string if item is number", () => {
            const input = 1017;
            const lst = ['123']
            const res = appendArrayOrNonArrayObjectToArray(lst, input);
            expect(res).toHaveLength(2);
            expect(res).toEqual(['123', '1017'])
        })

        test("If element in input is not string or number, it should not be pushed", () => {
            const input = ['MONDO:123', ['a', 'b']];
            const lst = ['123']
            const res = appendArrayOrNonArrayObjectToArray(lst, input);
            expect(res).toHaveLength(2);
            expect(res).toEqual(['123', 'MONDO:123'])
        })

        test("If input is string or array, it should not be pushed", () => {
            const input = { 'a': 'b' };
            const lst = ['123']
            const res = appendArrayOrNonArrayObjectToArray(lst, input);
            expect(res).toHaveLength(1);
            expect(res).toEqual(['123'])
        })

    })

    describe("Test generateObjectWithNoDuplicateElementsInValue function", () => {
        test("duplicate items are removed in object values", () => {
            const input = {
                item: ["123", "123", "345"]
            }
            expect(input.item).toHaveLength(3);
            const res = generateObjectWithNoDuplicateElementsInValue(input);
            expect(res.item).toHaveLength(2);
            expect(res.item).toEqual(['123', '345'])
        })
    })

    describe("Test generateDBIDs function", () => {
        test("id which is default curie should return itself", () => {
            const input = 'MONDO:000123';
            const res = generateDBID(input);
            expect(res).toEqual(input);
        })

        test("id which is not default curie should return prefixed removed version", () => {
            const input = 'NCBIGene:1017';
            const res = generateDBID(input);
            expect(res).toEqual('1017');
        })

        test("id which is not valid but has curie should return prefixed removed version", () => {
            const input = 'NCBIGene1:1017';
            const res = generateDBID(input);
            expect(res).toEqual('1017');
        })

        test("id which is not valid and is not curie should return itself", () => {
            const input = '1017';
            const res = generateDBID(input);
            expect(res).toEqual('1017');
        })

        test("id with multiple colons in it should only trim the chars before the first occurrence of colon", () => {
            const input = 'WIKIPATHWAYS:pathway:1017';
            const res = generateDBID(input);
            expect(res).toEqual('pathway:1017');
        })
    })

    describe("Test generateIDTypeDict function", () => {
        test("Test generateIDTypeDict function", () => {
            const res = generateIDTypeDict();
            expect(res['CHEBI']).toContain('SmallMolecule');
        })
    })
})