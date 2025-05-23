/**
 * Agent Toolkits
 * 
 * This example demonstrates how to use agent toolkits in LangChain:
 * 1. Pandas DataFrame Toolkit - For data analysis and manipulation
 * 2. SQL Database Toolkit - For database operations and queries
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { config } from "dotenv";
import { Logger } from "./Logger";
import { AIMessage, FunctionMessage, HumanMessage } from "@langchain/core/messages";
import axios from "axios";
import { routeResponse } from "./utils/routeResponse";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { AgentAction, AgentFinish } from "@langchain/core/agents";

// Load environment variables
config();

/**
 * Pandas DataFrame Toolkit
 * 
 * This toolkit provides tools for working with pandas DataFrames.
 * In a real implementation, you would use a proper DataFrame library.
 * For this example, we'll simulate DataFrame operations.
 */

// Define types for mock data
interface DataFrameRow {
  PassengerId: number;
  Survived: number;
  Pclass: number;
  Name: string;
  Sex: string;
  Age: number | null;
  SibSp: number;
  Parch: number;
  Ticket: string;
  Fare: number;
  Cabin: string | null;
  Embarked: string | null;
}

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

// Mock data
const mockData: DataFrameRow[] = [
  {
    PassengerId: 1,
    Survived: 1,
    Pclass: 3,
    Name: "Braund, Mr. Owen Harris",
    Sex: "male",
    Age: 22,
    SibSp: 1,
    Parch: 0,
    Ticket: "A/5 21171",
    Fare: 7.25,
    Cabin: null,
    Embarked: "S"
  },
  // ... add more mock data as needed
];

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

// Define prompts
const dfPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that analyzes data using pandas DataFrame operations."],
  ["user", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

const sqlPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant that analyzes data using SQL queries."],
  ["user", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Define passThrough for both agents
const passThrough = RunnablePassthrough.assign({
  agent_scratchpad: (input: Record<string, unknown>) => {
    const steps = (input.intermediate_steps as [AgentAction, string][]) || [];
    return steps.map(([action, observation]) => {
      const message = new AIMessage({
        content: "",
        additional_kwargs: {
          function_call: {
            name: action.tool,
            arguments: JSON.stringify(action.toolInput),
          },
        },
      });
      return message;
    });
  },
});

/**
 * SQL Database Toolkit
 * 
 * This toolkit provides tools for working with SQL databases.
 * For this example, we'll use a mock database with the Chinook schema.
 */

// Update tool types to use DynamicStructuredTool
const dfTools: DynamicStructuredTool[] = [
  new DynamicStructuredTool({
    name: "df_shape",
    description: "Get the shape of the DataFrame",
    schema: z.object({}),
    func: async () => {
      return JSON.stringify([mockData.length, Object.keys(mockData[0]).length]);
    },
  }),
  new DynamicStructuredTool({
    name: "df_head",
    description: "Get the first n rows of the DataFrame",
    schema: z.object({
      n: z.number().describe("Number of rows to return"),
    }),
    func: async ({ n }) => {
      const result = mockData.slice(0, n).map((row: DataFrameRow) => {
        const newRow: Record<string, any> = {};
        Object.keys(row).forEach(key => {
          newRow[key] = row[key as keyof DataFrameRow];
        });
        return newRow;
      });
      return JSON.stringify(result);
    },
  }),
  new DynamicStructuredTool({
    name: "df_describe",
    description: "Get statistical description of the DataFrame",
    schema: z.object({
      column: z.string().describe("Column name to describe"),
    }),
    func: async ({ column }) => {
      if (!mockData[0] || !(column in mockData[0])) {
        throw new Error(`Column ${column} not found in DataFrame`);
      }
      const values = mockData
        .map((row: DataFrameRow) => row[column as keyof DataFrameRow])
        .filter((val): val is number => typeof val === 'number');
      const stats = {
        count: values.length,
        mean: values.reduce((a, b) => a + b, 0) / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
      };
      return JSON.stringify(stats);
    },
  }),
];

const sqlTools: DynamicStructuredTool[] = [
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

/**
 * Agent Setup
 */
const chat = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-0125",
  temperature: 0,
});

/**
 * DataFrame Agent
 */
async function runDataFrameAgent(input: string) {
  const chain = RunnableSequence.from([
    passThrough,
    dfPrompt,
    chat.bind({ tools: dfTools }),
  ]);

  let intermediateSteps: [AgentAction, string][] = [];
  let chatHistory: (HumanMessage | AIMessage)[] = [];

  while (true) {
    const response = await chain.invoke({
      input,
      intermediate_steps: intermediateSteps,
      chat_history: chatHistory,
    });

    // If we got a direct response with content, return it
    if (response.content && !response.tool_calls?.length) {
      chatHistory.push(new HumanMessage(input));
      chatHistory.push(new AIMessage(String(response.content)));
      return response.content;
    }

    // If we got a final response with returnValues, return it
    if ('returnValues' in response && typeof response.returnValues === 'object' && response.returnValues !== null) {
      const returnValues = response.returnValues as { output: string };
      chatHistory.push(new HumanMessage(input));
      chatHistory.push(new AIMessage(returnValues.output));
      return returnValues.output;
    }

    // Get tool name and args from the response
    const toolName = response.tool_calls?.[0]?.name;
    const toolArgs = response.tool_calls?.[0]?.args;

    // Validate that we have a valid tool call
    if (!toolName || !toolArgs) {
      throw new Error(`Invalid tool call response: ${JSON.stringify(response)}`);
    }

    // Create the agent action
    const action: AgentAction = {
      tool: toolName,
      toolInput: toolArgs,
      log: typeof response.content === 'string' ? response.content : ''
    };

    const result = await routeResponse(response, dfTools);
    intermediateSteps.push([action, result]);
  }
}

/**
 * SQL Agent
 */
async function runSQLAgent(input: string) {
  const chain = RunnableSequence.from([
    passThrough,
    sqlPrompt,
    chat.bind({ tools: sqlTools }),
  ]);

  let intermediateSteps: [AgentAction, string][] = [];
  let chatHistory: (HumanMessage | AIMessage)[] = [];

  while (true) {
    const response = await chain.invoke({
      input,
      intermediate_steps: intermediateSteps,
      chat_history: chatHistory,
    });

    // If we got a direct response with content, return it
    if (response.content && !response.tool_calls?.length) {
      chatHistory.push(new HumanMessage(input));
      chatHistory.push(new AIMessage(String(response.content)));
      return response.content;
    }

    // If we got a final response with returnValues, return it
    if ('returnValues' in response && typeof response.returnValues === 'object' && response.returnValues !== null) {
      const returnValues = response.returnValues as { output: string };
      chatHistory.push(new HumanMessage(input));
      chatHistory.push(new AIMessage(returnValues.output));
      return returnValues.output;
    }

    // Get tool name and args from the response
    const toolName = response.tool_calls?.[0]?.name;
    const toolArgs = response.tool_calls?.[0]?.args;

    // Validate that we have a valid tool call
    if (!toolName || !toolArgs) {
      throw new Error(`Invalid tool call response: ${JSON.stringify(response)}`);
    }

    // Create the agent action
    const action: AgentAction = {
      tool: toolName,
      toolInput: toolArgs,
      log: typeof response.content === 'string' ? response.content : ''
    };

    const result = await routeResponse(response, sqlTools);
    intermediateSteps.push([action, result]);
  }
}

/**
 * Example Usage
 */
async function main() {
  try {
    Logger.info("Starting agent toolkits example\n");

    // Test DataFrame Agent
    Logger.info("Testing DataFrame Agent\n");
    const dfResponse = await runDataFrameAgent("What is the shape of the DataFrame?");
    Logger.success("DataFrame Agent response received\n");
    Logger.debug(`DataFrame Agent response: ${dfResponse}\n`);

    // Test SQL Agent
    Logger.info("Testing SQL Agent\n");
    const sqlResponse = await runSQLAgent("Which artist has the most albums?");
    Logger.success("SQL Agent response received\n");
    Logger.debug(`SQL Agent response: ${sqlResponse}\n`);

  } catch (error) {
    Logger.error(`Error in main: ${error}\n`);
  }
}

// Run the example
main(); 