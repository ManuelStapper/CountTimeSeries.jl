using Documenter, CountTimeSeries

push!(LOAD_PATH,"../src/")

makedocs(sitename = "CountTimeSeries.jl", pages = [
    "Home" => "Home.md",
    "Getting Started" => "GettingStarted.md",
    "Theoretical Background" => ["INGARCH Framework" => "INGARCH.md", "INARMA Framework" => "INARMA.md"],
    "Application" => "Application.md",
    "Outlook" => "Outlook.md",
    "Types, Structs and Functions" => "TypesStructsFunctions.md"
])
