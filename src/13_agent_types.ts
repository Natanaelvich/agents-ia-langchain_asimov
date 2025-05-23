/**
 * Agent Types
 * 
 * This example demonstrates different types of agents in LangChain:
 * 1. Tool Calling Agent - Uses structured tool calling for more reliable tool execution
 * 2. ReAct Agent - Uses a reasoning and acting pattern for more complex problem solving
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { config } from "dotenv";
import { Logger } from "./Logger";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentAction, AgentFinish } from "@langchain/core/agents";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { AIMessage, FunctionMessage, HumanMessage } from "@langchain/core/messages";
import { routeResponse } from "./utils/routeResponse";

// Load environment variables
config();

/**
 * Tool Schemas
 */
const PythonReplSchema = z.object({
  query: z.string().describe("The Python code to execute"),
});

type PythonReplInput = z.infer<typeof PythonReplSchema>;

/**
 * Python REPL Tool
 * 
 * This tool allows the agent to execute Python code and get results.
 * In a real implementation, this would need proper sandboxing and security measures.
 */
const pythonReplTool = tool(
  async (input: PythonReplInput): Promise<string> => {
    try {
      // In a real implementation, this would execute the Python code in a safe environment
      // For this example, we'll just return a mock response
      if (input.query.includes("fibonacci")) {
        return "55"; // Mock response for fibonacci(10)
      } else if (input.query.includes("len")) {
        return "9"; // Mock response for len("LangChain")
      }
      return "Code executed successfully";
    } catch (error) {
      Logger.error(`Error executing Python code: ${error}`);
      throw error;
    }
  },
  {
    name: "python_repl_ast",
    description: "Executes Python code and returns the result",
    schema: PythonReplSchema,
  }
);

const tools = [pythonReplTool];

/**
 * Tool Calling Agent Setup
 * 
 * This agent type uses structured tool calling for more reliable tool execution.
 * It's designed to work with the OpenAI function calling format.
 */
const toolCallingPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a friendly assistant. Use the Python REPL tool to help answer questions."],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

const chat = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-0125",
  temperature: 0,
});

/**
 * ReAct Agent Setup
 * 
 * This agent type uses a reasoning and acting pattern:
 * 1. Thought: The agent thinks about what to do
 * 2. Action: The agent decides which tool to use
 * 3. Action Input: The agent provides input for the tool
 * 4. Observation: The agent observes the result
 * 5. Repeat until final answer
 */
const reactPrompt = ChatPromptTemplate.fromTemplate(
  `Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}`
);

/**
 * Agent Chain Setup
 */
interface AgentInput {
  input: string;
  intermediate_steps: [AgentAction, string][];
  chat_history?: (HumanMessage | AIMessage)[];
  tools?: string;
  tool_names?: string;
}

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
 * Tool Calling Agent Executor
 */
async function runToolCallingAgent(input: string) {
  const chain = RunnableSequence.from([
    passThrough,
    toolCallingPrompt,
    chat.bind({ tools }),
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

    const result = await routeResponse(response, tools);
    intermediateSteps.push([action, result]);
  }
}

/**
 * ReAct Agent Executor
 */
async function runReActAgent(input: string) {
  const chain = RunnableSequence.from([
    passThrough,
    reactPrompt,
    chat.bind({ tools }),
  ]);

  let intermediateSteps: [AgentAction, string][] = [];
  let chatHistory: (HumanMessage | AIMessage)[] = [];

  while (true) {
    const response = await chain.invoke({
      input,
      intermediate_steps: intermediateSteps,
      chat_history: chatHistory,
      tools: tools.map(tool => `${tool.name}: ${tool.description}`).join('\n'),
      tool_names: tools.map(tool => tool.name).join(', '),
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

    const result = await routeResponse(response, tools);
    intermediateSteps.push([action, result]);
  }
}

/**
 * Example Usage
 */
async function main() {
  try {
    Logger.info("Starting agent types example\n");

    // Test Tool Calling Agent
    Logger.info("Testing Tool Calling Agent\n");
    const toolCallingResponse = await runToolCallingAgent("Qual é o décimo valor da sequência fibonacci?");
    Logger.success("Tool Calling Agent response received\n");
    Logger.debug(`Tool Calling Agent response: ${toolCallingResponse}\n`);

    // Test ReAct Agent
    Logger.info("Testing ReAct Agent\n");
    const reactResponse = await runReActAgent("Quantas letras tem a palavra LangChain?");
    Logger.success("ReAct Agent response received\n");
    Logger.debug(`ReAct Agent response: ${reactResponse}\n`);

  } catch (error) {
    Logger.error(`Error in main: ${error}\n`);
  }
}

// Run the example
main(); 