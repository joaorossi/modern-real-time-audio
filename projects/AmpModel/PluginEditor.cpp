#include "PluginProcessor.h"
#include "PluginEditor.h"

AmpModelProcessorEditor::AmpModelProcessorEditor(AmpModelProcessor& p) :
    AudioProcessorEditor(&p), audioProcessor(p),
    genericParameterEditor(audioProcessor.getParameterManager())
{
    int height = static_cast<int>(audioProcessor.getParameterManager().getParameters().size())
               * genericParameterEditor.parameterWidgetHeight;
    setSize(300, height);
    addAndMakeVisible(genericParameterEditor);
}

AmpModelProcessorEditor::~AmpModelProcessorEditor()
{
}

void AmpModelProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void AmpModelProcessorEditor::resized()
{
    genericParameterEditor.setBounds(getLocalBounds());
}
