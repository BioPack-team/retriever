import RedisMock from 'ioredis-mock';

describe('Test redis client', () => {
  const OLD_ENV = process.env;

  beforeEach(() => {
    jest.resetModules(); // Most important - it clears the cache
    jest.clearAllMocks();
    jest.mock('ioredis', () => RedisMock);
    process.env = { ...OLD_ENV }; // Make a copy
  });

  afterAll(() => {
    process.env = OLD_ENV; // Restore old environment
  });
  test('Client reports not enabled when REDIS_HOST is not set', () => {
    const { redisClient } = require('../../src/redis-client');
    expect(redisClient.clientEnabled).toEqual(false);
  });

  test('will receive process.env variables', () => {
    // Set the variables
    process.env.REDIS_HOST = 'localhost';
    process.env.REDIS_PORT = '3367';
    const { redisClient } = require('../../src/redis-client');
    expect(redisClient).not.toEqual({});
  });

  test('Test if record is correctly stored', async () => {
    process.env.REDIS_HOST = 'localhost';
    process.env.REDIS_PORT = '3367';
    const { redisClient } = require('../../src/redis-client');
    await redisClient.client.setTimeout('record1', 'hello');
    const res = await redisClient.client.getTimeout('record1');
    expect(res).toEqual('hello');
  });

  test('Test if record is correctly stored', async () => {
    process.env.REDIS_HOST = 'localhost';
    process.env.REDIS_PORT = '3367';
    const { redisClient } = require('../../src/redis-client');
    await redisClient.client.setTimeout('record1', 'hello');
    const res = await redisClient.client.getTimeout('record1');
    expect(res).toEqual('hello');
  });

  test('Test key should be removed after ttl', async () => {
    process.env.REDIS_HOST = 'localhost';
    process.env.REDIS_PORT = '3367';
    const { redisClient } = require('../../src/redis-client');
    await redisClient.client.setTimeout('record1', 'hello', 'EX', 2);
    await new Promise((r) => setTimeout(r, 3000));
    const res = await redisClient.client.getTimeout('record1');
    expect(res).not.toBeNull;
  });
});
