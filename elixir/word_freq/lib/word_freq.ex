defmodule WordFreq do
  @moduledoc """
  Documentation for `WordFreq`.
  """

  @doc """
  """
  def run do
    path = "priv/sample.txt"
    case File.read(path) do
      {:ok, content} -> 
        IO.puts(content)
      {:error, reason} ->
    end
    # PATH |> File.read |> IO.puts
  end
end
