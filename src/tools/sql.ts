import { DynamicStructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { Logger } from "../Logger";

/**
 * Define types for mock database
 */
interface MockTable {
  schema: {
    columns: string[];
    types: string[];
  };
  data: Record<string, any>[];
}

interface MockDatabase {
  [key: string]: MockTable;
}

// Mock database
const mockTables: MockDatabase = {
  Album: {
    schema: {
      columns: ["AlbumId", "Title", "ArtistId"],
      types: ["INTEGER", "NVARCHAR(160)", "INTEGER"]
    },
    data: [
      { AlbumId: 1, Title: "For Those About To Rock We Salute You", ArtistId: 1 },
      { AlbumId: 2, Title: "Balls to the Wall", ArtistId: 2 }
    ]
  },
  Artist: {
    schema: {
      columns: ["ArtistId", "Name"],
      types: ["INTEGER", "NVARCHAR(120)"]
    },
    data: [
      { ArtistId: 1, Name: "AC/DC" },
      { ArtistId: 2, Name: "Accept" }
    ]
  }
};

/**
 * SQL Database Tools
 */
export const sqlTools: DynamicStructuredTool[] = [
  new DynamicStructuredTool({
    name: "list_tables",
    description: "List all tables in the database",
    schema: z.object({}),
    func: async () => {
      return JSON.stringify(Object.keys(mockTables));
    },
  }),
  new DynamicStructuredTool({
    name: "get_table_schema",
    description: "Get the schema of a table",
    schema: z.object({
      table: z.string().describe("Table name"),
    }),
    func: async ({ table }) => {
      if (!mockTables[table]) {
        throw new Error(`Table ${table} not found`);
      }
      return JSON.stringify(mockTables[table].schema);
    },
  }),
  new DynamicStructuredTool({
    name: "run_query",
    description: "Run a SQL query on the database",
    schema: z.object({
      query: z.string().describe("SQL query to execute"),
    }),
    func: async ({ query }) => {
      // Mock query execution
      if (query.toLowerCase().includes('select * from album')) {
        return JSON.stringify(mockTables.Album.data);
      }
      if (query.toLowerCase().includes('select * from artist')) {
        return JSON.stringify(mockTables.Artist.data);
      }
      throw new Error(`Query not supported: ${query}`);
    },
  }),
]; 