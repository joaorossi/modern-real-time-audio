#include "PluginProcessor.h"
#include "PluginEditor.h"

NeuralNetworksProcessorEditor::NeuralNetworksProcessorEditor(NeuralNetworksProcessor& p) :
    AudioProcessorEditor(&p), audioProcessor(p),
    genericParameterEditor(audioProcessor.getParameterManager())
{
    int height = static_cast<int>(audioProcessor.getParameterManager().getParameters().size())
               * genericParameterEditor.parameterWidgetHeight;
    setSize(300, height);
    addAndMakeVisible(genericParameterEditor);
}

NeuralNetworksProcessorEditor::~NeuralNetworksProcessorEditor()
{
}

void NeuralNetworksProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void NeuralNetworksProcessorEditor::resized()
{
    genericParameterEditor.setBounds(getLocalBounds());
}
